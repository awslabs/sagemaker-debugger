# First Party
# Third Party
import pytest

from smdebug.profiler.analysis.utils.pandas_data_analysis import (
    PandasFrameAnalysis,
    Resource,
    StatsBy,
)
from smdebug.profiler.analysis.utils.profiler_data_to_pandas import PandasFrame


@pytest.mark.parametrize("framework", ["tf2", "pt"])
def test_pandas_frames(framework):
    bucket_name = (
        "s3://smdebug-testing/resources/" + framework + "_detailed_profile/profiler-output"
    )
    pf = PandasFrame(bucket_name, scan_interval=50000000000)
    system_metrics_df = pf.get_all_system_metrics()

    print(f"Number of rows in system metrics dataframe = {system_metrics_df.shape[0]}")
    if framework == "tf2":
        assert system_metrics_df.shape[0] == 39392
    if framework == "pt":
        assert system_metrics_df.shape[0] == 84768
    print(f"Number of columns in system metrics dataframe = {system_metrics_df.shape[1]}")
    if framework == "tf2":
        assert system_metrics_df.shape[1] == 7
    if framework == "pt":
        assert system_metrics_df.shape[1] == 7

    pf = PandasFrame(bucket_name, scan_interval=50000000000)
    framework_metrics_df = pf.get_all_framework_metrics()

    print(f"Number of rows in framework metrics dataframe = {framework_metrics_df.shape[0]}")
    if framework == "tf2":
        assert framework_metrics_df.shape[0] == 73984
    if framework == "pt":
        assert framework_metrics_df.shape[0] == 154192
    print(f"Number of columns in framework metrics dataframe = {framework_metrics_df.shape[1]}")
    if framework == "tf2":
        assert framework_metrics_df.shape[1] == 11
    if framework == "pt":
        assert framework_metrics_df.shape[1] == 11


@pytest.mark.parametrize("framework", ["tf2", "pt"])
def test_get_data_by_time(framework):
    bucket_name = (
        "s3://smdebug-testing/resources/" + framework + "_detailed_profile/profiler-output"
    )
    pf = PandasFrame(bucket_name, scan_interval=50000000000)
    if framework == "tf2":
        system_metrics_df, framework_metrics_df = pf.get_profiler_data_by_time(
            1596668220000000, 1596678220000000
        )
        assert system_metrics_df.shape[0] == 39392
    if framework == "pt":
        system_metrics_df, framework_metrics_df = pf.get_profiler_data_by_time(
            1596493560000000, 1596499560000000
        )
        assert system_metrics_df.shape[0] == 84768
    print(f"Number of rows in system metrics dataframe = {system_metrics_df.shape[0]}")

    print(f"Number of columns in system metrics dataframe = {system_metrics_df.shape[1]}")
    assert system_metrics_df.shape[1] == 7

    print(f"Number of rows in framework metrics dataframe = {framework_metrics_df.shape[0]}")
    if framework == "tf2":
        assert framework_metrics_df.shape[0] == 73984
    if framework == "pt":
        assert framework_metrics_df.shape[0] == 154192
    print(f"Number of columns in framework metrics dataframe = {framework_metrics_df.shape[1]}")
    assert framework_metrics_df.shape[1] == 11


@pytest.mark.parametrize("framework", ["tf2", "pt"])
def test_get_data_by_step(framework):
    bucket_name = (
        "s3://smdebug-testing/resources/" + framework + "_detailed_profile/profiler-output"
    )
    pf = PandasFrame(bucket_name)
    _, framework_metrics_df = pf.get_profiler_data_by_step(2, 3)

    assert not framework_metrics_df.empty

    assert framework_metrics_df.groupby("step").ngroups == 2

    print(f"Number of rows in framework metrics dataframe = {framework_metrics_df.shape[0]}")
    if framework == "tf2":
        assert framework_metrics_df.shape[0] == 5
    if framework == "pt":
        assert framework_metrics_df.shape[0] == 738

    print(f"Number of columns in framework metrics dataframe = {framework_metrics_df.shape[1]}")
    assert framework_metrics_df.shape[1] == 11


def get_metrics(framework):
    bucket_name = (
        "s3://smdebug-testing/resources/" + framework + "_detailed_profile/profiler-output"
    )
    pf = PandasFrame(bucket_name, use_in_memory_cache=True)
    system_metrics_df, framework_metrics_df = (
        pf.get_all_system_metrics(),
        pf.get_all_framework_metrics(),
    )
    return system_metrics_df, framework_metrics_df


@pytest.fixture(scope="module", autouse=True)
def tf_pandas_frame_analysis():
    return PandasFrameAnalysis(*get_metrics("tf2"))


@pytest.fixture(scope="module", autouse=True)
def pt_pandas_frame_analysis():
    return PandasFrameAnalysis(*get_metrics("pt"))


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
@pytest.mark.parametrize(
    "by", [StatsBy.TRAINING_PHASE, StatsBy.FRAMEWORK_METRICS, StatsBy.PROCESS, "step"]
)
def test_get_step_stats(framework, by, tf_pandas_frame_analysis, pt_pandas_frame_analysis):
    if framework == "tf2":
        pf_analysis = tf_pandas_frame_analysis
    else:
        pf_analysis = pt_pandas_frame_analysis

    step_stats = pf_analysis.get_step_statistics(by=by)

    if by == "step":
        assert step_stats is None
    else:
        assert not step_stats.empty
        assert step_stats.shape[1] == 7

        if by == "training_phase":
            if framework == "tf2":
                assert step_stats.shape[0] == 2
            else:
                assert step_stats.shape[0] == 1
        elif by == "framework_metric":
            if framework == "tf2":
                assert step_stats.shape[0] == 111
            else:
                assert step_stats.shape[0] == 207
        elif by == "process":
            if framework == "tf2":
                assert step_stats.shape[0] == 6
            else:
                assert step_stats.shape[0] == 7


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
@pytest.mark.parametrize("phase", [["Step:ModeKeys.TRAIN"], None])
def test_get_util_stats_by_training_phase(
    framework, phase, tf_pandas_frame_analysis, pt_pandas_frame_analysis
):
    if framework == "tf2":
        pf_analysis = tf_pandas_frame_analysis
    else:
        if phase and phase[0] == "Step:ModeKeys.TRAIN":
            phase = ["Step:ModeKeys.GLOBAL"]
        pf_analysis = pt_pandas_frame_analysis

    util_stats = pf_analysis.get_utilization_stats(phase=phase, by=StatsBy.TRAINING_PHASE)

    assert not util_stats.empty
    if phase is None:
        assert util_stats.shape[0] <= 8
    else:
        assert util_stats.shape[0] <= 8
    assert util_stats.shape[1] == 9

    assert all(util_stats["Resource"].unique() == ["cpu", "gpu"])


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
@pytest.mark.parametrize("resource", [None, Resource.CPU, [Resource.CPU, Resource.GPU], "cpu"])
@pytest.mark.parametrize("by", [None, "step"])
def test_get_util_stats(
    framework, resource, by, tf_pandas_frame_analysis, pt_pandas_frame_analysis
):
    if framework == "tf2":
        pf_analysis = tf_pandas_frame_analysis
    else:
        pf_analysis = pt_pandas_frame_analysis

    util_stats = pf_analysis.get_utilization_stats(resource=resource, by=by)

    if by == "step" or resource == "cpu":
        assert util_stats is None
    else:
        assert not util_stats.empty


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
@pytest.mark.parametrize("device", ["cpu", Resource.CPU, Resource.GPU])
@pytest.mark.parametrize(
    "ranges", [None, [(0, 10), (10, 20), (30, 80), (80, 100)], [(30,)], [], ((0, 10), (10, 90))]
)
def test_get_device_usage_stats(
    framework, device, ranges, tf_pandas_frame_analysis, pt_pandas_frame_analysis
):
    if framework == "tf2":
        pf_analysis = tf_pandas_frame_analysis
    else:
        pf_analysis = pt_pandas_frame_analysis

    usage_stats = pf_analysis.get_device_usage_stats(device=device, utilization_ranges=ranges)

    if ranges in [[(30,)], [], ((0, 10), (10, 90))] or device == "cpu":
        assert usage_stats.empty
    else:
        assert not usage_stats.empty

        if ranges is None:
            assert usage_stats.shape[1] <= 3 + 3
        else:
            assert usage_stats.shape[1] <= 3 + len(ranges)


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
@pytest.mark.parametrize(
    "phase",
    [
        ["Step:ModeKeys.TRAIN"],
        "Step:ModeKeys.TRAIN",
        ["Step:ModeKeys.GLOBAL"],
        ("Step:ModeKeys.GLOBAL"),
    ],
)
def test_get_training_phase_intervals(
    framework, phase, tf_pandas_frame_analysis, pt_pandas_frame_analysis
):
    if framework == "tf2":
        valid_phase = ["Step:ModeKeys.TRAIN"]
        pf_analysis = tf_pandas_frame_analysis
    else:
        valid_phase = ["Step:ModeKeys.GLOBAL"]
        pf_analysis = pt_pandas_frame_analysis

    interval_stats = pf_analysis.get_training_phase_intervals(phase=phase)

    if isinstance(phase, str):
        phase = [phase]
    if not isinstance(phase, list) or phase != valid_phase:
        print(not isinstance(phase, (str, list)))
        print(phase != valid_phase, phase, valid_phase)
        print((isinstance(phase, str) and [phase] != valid_phase))
        assert interval_stats is None
    else:
        assert not interval_stats.empty
        assert interval_stats.shape[1] == 3

        if framework == "tf2":
            assert interval_stats.shape[0] == 11251
        else:
            assert interval_stats.shape[0] == 785


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2"])
def test_get_jobs_stats(framework, tf_pandas_frame_analysis, pt_pandas_frame_analysis):
    if framework == "tf2":
        pf_analysis = tf_pandas_frame_analysis
    else:
        pf_analysis = pt_pandas_frame_analysis

    job_stats = pf_analysis.get_job_statistics()
    assert job_stats is not None
