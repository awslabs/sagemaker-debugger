# First Party
# Third Party
import pytest

from smdebug.profiler.analysis.utils.pandas_data_analysis import PandasFrameAnalysis
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
        assert system_metrics_df.shape[1] == 4
    if framework == "pt":
        assert system_metrics_df.shape[1] == 4

    pf = PandasFrame(bucket_name, scan_interval=50000000000)
    framework_metrics_df = pf.get_all_framework_metrics()

    print(f"Number of rows in framework metrics dataframe = {framework_metrics_df.shape[0]}")
    if framework == "tf2":
        assert framework_metrics_df.shape[0] == 74001
    if framework == "pt":
        assert framework_metrics_df.shape[0] == 154192
    print(f"Number of columns in framework metrics dataframe = {framework_metrics_df.shape[1]}")
    if framework == "tf2":
        assert framework_metrics_df.shape[1] == 10
    if framework == "pt":
        assert framework_metrics_df.shape[1] == 10


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
    assert system_metrics_df.shape[1] == 4

    print(f"Number of rows in framework metrics dataframe = {framework_metrics_df.shape[0]}")
    if framework == "tf2":
        assert framework_metrics_df.shape[0] == 74001
    if framework == "pt":
        assert framework_metrics_df.shape[0] == 154192
    print(f"Number of columns in framework metrics dataframe = {framework_metrics_df.shape[1]}")
    assert framework_metrics_df.shape[1] == 10


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
    assert framework_metrics_df.shape[1] == 10


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
def tensorflow_metrics():
    return get_metrics("tf2")


@pytest.fixture(scope="module", autouse=True)
def pytorch_metrics():
    return get_metrics("pt")


@pytest.fixture
def phase_train():
    return ["Step:ModeKeys.TRAIN"]


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
@pytest.mark.parametrize("by", ["training_phase", "framework_metric", "process"])
def test_get_step_stats(framework, by, tensorflow_metrics, pytorch_metrics):
    if framework == "tf2":
        system_metrics_df, framework_metrics_df = tensorflow_metrics
    else:
        system_metrics_df, framework_metrics_df = pytorch_metrics

    pf_analysis = PandasFrameAnalysis(system_metrics_df, framework_metrics_df)
    step_stats = pf_analysis.get_step_statistics(by=by)
    print(step_stats.shape)
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
def test_get_util_stats_by_training_phase(framework, phase, tensorflow_metrics, pytorch_metrics):
    if framework == "tf2":
        system_metrics_df, framework_metrics_df = tensorflow_metrics
    else:
        if phase and phase[0] == "Step:ModeKeys.TRAIN":
            phase = ["Step:ModeKeys.GLOBAL"]
        system_metrics_df, framework_metrics_df = pytorch_metrics

    pf_analysis = PandasFrameAnalysis(system_metrics_df, framework_metrics_df)
    util_stats = pf_analysis.get_utilization_stats(phase=phase, by="training_phase")

    assert not util_stats.empty
    if phase is None:
        assert util_stats.shape[0] <= 8
    else:
        assert util_stats.shape[0] <= 8
    assert util_stats.shape[1] == 8

    assert all(util_stats["Resource"].unique() == ["cpu", "gpu"])


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
def test_get_util_stats(framework, tensorflow_metrics, pytorch_metrics):
    if framework == "tf2":
        system_metrics_df, framework_metrics_df = tensorflow_metrics
    else:
        system_metrics_df, framework_metrics_df = pytorch_metrics

    pf_analysis = PandasFrameAnalysis(system_metrics_df, framework_metrics_df)
    util_stats = pf_analysis.get_utilization_stats()

    assert not util_stats.empty


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("ranges", [None, [(0, 10), (10, 20), (30, 80), (80, 100)]])
def test_get_device_usage_stats(framework, device, ranges, tensorflow_metrics, pytorch_metrics):
    if framework == "tf2":
        system_metrics_df, framework_metrics_df = tensorflow_metrics
    else:
        system_metrics_df, framework_metrics_df = pytorch_metrics

    pf_analysis = PandasFrameAnalysis(system_metrics_df, framework_metrics_df)
    usage_stats = pf_analysis.get_device_usage_stats(device=device, utilization_ranges=ranges)

    assert usage_stats

    if ranges is None:
        assert len(usage_stats) == 3
    else:
        assert len(usage_stats) == len(ranges)


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
def test_get_training_phase_intervals(framework, phase_train, tensorflow_metrics, pytorch_metrics):
    if framework == "tf2":
        system_metrics_df, framework_metrics_df = tensorflow_metrics
    else:
        phase_train = ["Step:ModeKeys.GLOBAL"]
        system_metrics_df, framework_metrics_df = pytorch_metrics

    pf_analysis = PandasFrameAnalysis(system_metrics_df, framework_metrics_df)
    interval_stats = pf_analysis.get_training_phase_intervals(phase=phase_train)

    assert not interval_stats.empty
    assert interval_stats.shape[1] == 3

    if framework == "tf2":
        assert interval_stats.shape[0] == 11251
    else:
        assert interval_stats.shape[0] == 2793


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["tf2", "pt"])
def test_get_jobs_stats(framework, tensorflow_metrics, pytorch_metrics):
    if framework == "tf2":
        system_metrics_df, framework_metrics_df = tensorflow_metrics
    else:
        system_metrics_df, framework_metrics_df = pytorch_metrics

    pf_analysis = PandasFrameAnalysis(system_metrics_df, framework_metrics_df)

    job_stats = pf_analysis.get_job_statistics()
    assert job_stats is not None
