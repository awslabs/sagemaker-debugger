# Standard Library
import json
import os
import pstats
import shutil
import time

# Third Party
import boto3
import pytest

# First Party
from smdebug.core.access_layer.utils import is_s3
from smdebug.profiler.analysis.python_profile_analysis import PyinstrumentAnalysis, cProfileAnalysis
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profiler import (
    PyinstrumentPythonProfiler,
    StepPhase,
    cProfilePythonProfiler,
)


@pytest.fixture
def test_framework():
    return "test-framework"


@pytest.fixture
def cprofile_python_profiler(out_dir, test_framework):
    return cProfilePythonProfiler(out_dir, test_framework)


@pytest.fixture
def pyinstrument_python_profiler(out_dir, test_framework):
    return PyinstrumentPythonProfiler(out_dir, test_framework)


@pytest.fixture
def cprofile_dir(out_dir, test_framework):
    return "{0}/framework/{1}/{2}".format(out_dir, test_framework, CPROFILE_NAME)


@pytest.fixture
def pyinstrument_dir(out_dir, test_framework):
    return "{0}/framework/{1}/{2}".format(out_dir, test_framework, PYINSTRUMENT_NAME)


@pytest.fixture(autouse=True)
def reset_python_profiler_dir(cprofile_dir, pyinstrument_dir):
    shutil.rmtree(cprofile_dir, ignore_errors=True)
    shutil.rmtree(pyinstrument_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def bucket_prefix():
    return f"s3://smdebug-testing/resources/python_profile/{int(time.time())}"


@pytest.fixture(scope="session")
def sample_times():
    return [time.time() for _ in range(5)]


def _profiling_set_up(python_profiler, start_step, end_step):
    current_step = start_step

    while current_step < end_step:
        python_profiler.start_profiling(StepPhase.STEP_START, start_step=current_step)
        assert python_profiler._start_step == current_step
        assert python_profiler._start_phase == StepPhase.STEP_START
        python_profiler.stop_profiling(StepPhase.STEP_END, current_step)
        current_step += 1


def _analysis_set_up(python_profiler):
    def start_end_step_function():
        time.sleep(
            0.0011
        )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds

    def end_start_step_function():
        time.sleep(
            0.0011
        )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds

    def time_function():
        time.sleep(
            0.0011
        )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds

    # start_end_step_function is called in between the start and end of the first step.
    python_profiler.start_profiling(StepPhase.STEP_START, 1)
    start_end_step_function()
    python_profiler.stop_profiling(StepPhase.STEP_END, 1)

    # end_start_step_function is called in between the end of the first step and the start of the second step.
    python_profiler.start_profiling(StepPhase.STEP_END, 1)
    end_start_step_function()
    python_profiler.stop_profiling(StepPhase.STEP_START, 2)

    # time function is called in between the start and end of the second step.
    python_profiler.start_profiling(StepPhase.STEP_START, 2)
    time_function()
    python_profiler.stop_profiling(StepPhase.STEP_END, 2)


def _upload_s3_folder(bucket, key, folder):
    s3_client = boto3.client("s3")
    for root, _, files in os.walk(folder):
        for file in files:
            dir = os.path.basename(root)
            full_key = os.path.join(key, dir, file)
            s3_client.upload_file(os.path.join(root, file), bucket, full_key)


@pytest.mark.parametrize("steps", [(1, 2), (1, 5)])
def test_cprofile_profiling(cprofile_python_profiler, steps, cprofile_dir):
    """
    This test is meant to test that profiling with cProfile produces the correct number of stats files based on the
    step range of profiling and that each of those files is in a valid format.
    """
    start_step, end_step = steps
    _profiling_set_up(cprofile_python_profiler, start_step, end_step)

    # Test that directory and corresponding files exist.
    assert os.path.isdir(cprofile_dir)

    stats_dirs = os.listdir(cprofile_dir)
    assert len(stats_dirs) == (end_step - start_step)
    for stats_dir in stats_dirs:
        full_stats_path = os.path.join(cprofile_dir, stats_dir, CPROFILE_STATS_FILENAME)
        assert os.path.isfile(full_stats_path)
        assert pstats.Stats(full_stats_path)  # validate output file


@pytest.mark.parametrize("steps", [(1, 2), (1, 5)])
def test_pyinstrument_profiling(pyinstrument_python_profiler, steps, pyinstrument_dir):
    """
    This test is meant to test that profiling with pyinstrument produces the correct number of stats files based on the
    step range of profiling and that each of those files is in a valid format.
    """
    start_step, end_step = steps
    _profiling_set_up(pyinstrument_python_profiler, start_step, end_step)

    # Test that directory and corresponding files exist.
    assert os.path.isdir(pyinstrument_dir)

    stats_dirs = os.listdir(pyinstrument_dir)
    assert len(stats_dirs) == (end_step - start_step)
    for stats_dir in stats_dirs:
        full_stats_path = os.path.join(pyinstrument_dir, stats_dir, PYINSTRUMENT_JSON_FILENAME)
        assert os.path.isfile(full_stats_path)
        with open(full_stats_path, "r") as f:
            assert json.load(f)  # validate output file


@pytest.mark.parametrize("s3", [False, True])
def test_cprofile_analysis(
    cprofile_python_profiler, cprofile_dir, bucket_prefix, test_framework, s3
):
    """
    This test is meant to test that the cProfile analysis retrieves the correct step's stats based on the specified
    interval. Stats are either retrieved from s3 or generated manually through python profiling.
    """
    if s3:
        # Fetch stats from s3
        os.makedirs(cprofile_dir)
        python_profile_analysis = cProfileAnalysis(
            local_profile_dir=cprofile_dir, s3_path=bucket_prefix
        )
    else:
        # Do analysis and use those stats.
        _analysis_set_up(cprofile_python_profiler)
        python_profile_analysis = cProfileAnalysis(local_profile_dir=cprofile_dir)
        _, bucket, prefix = is_s3(bucket_prefix)
        key = os.path.join(prefix, "framework", test_framework, CPROFILE_NAME)
        _upload_s3_folder(bucket, key, cprofile_dir)

    assert len(python_profile_analysis.python_profile_stats) == 3

    # Test that start_end_step_function call is recorded in received stats, but not end_start_step_function or
    # time_function.
    stats = python_profile_analysis.fetch_profile_stats_by_step(1)
    function_stats_list = stats.function_stats_list
    assert len(function_stats_list) > 0
    assert any(["start_end_step_function" in stat.function_name for stat in function_stats_list])
    assert all(
        ["end_start_step_function" not in stat.function_name for stat in function_stats_list]
    )
    assert all(["time_function" not in stat.function_name for stat in function_stats_list])

    # Test that end_start_step_function call is recorded in received stats, but not start_end_step_function or
    # time_function.
    stats = python_profile_analysis.fetch_profile_stats_by_step(
        1, end_step=2, start_phase=StepPhase.STEP_END, end_phase=StepPhase.STEP_START
    )
    function_stats_list = stats.function_stats_list
    assert len(function_stats_list) > 0
    assert all(
        ["start_end_step_function" not in stat.function_name for stat in function_stats_list]
    )
    assert any(["end_start_step_function" in stat.function_name for stat in function_stats_list])
    assert all(["time_function" not in stat.function_name for stat in function_stats_list])

    # Test that time_function call is recorded in received stats, but not start_end_step_function or
    # end_start_step_function
    time_function_step_stats = python_profile_analysis.python_profile_stats[-1]
    step_start_time = (
        time_function_step_stats.start_time_since_epoch_in_micros / CONVERT_TO_MICROSECS
    )
    stats = python_profile_analysis.fetch_profile_stats_by_time(step_start_time, time.time())
    function_stats_list = stats.function_stats_list
    assert len(function_stats_list) > 0
    assert all(
        ["start_end_step_function" not in stat.function_name for stat in function_stats_list]
    )
    assert all(
        ["end_start_step_function" not in stat.function_name for stat in function_stats_list]
    )
    assert any(["time_function" in stat.function_name for stat in function_stats_list])


@pytest.mark.parametrize("s3", [False, True])
def test_pyinstrument_analysis(
    pyinstrument_python_profiler, pyinstrument_dir, test_framework, bucket_prefix, s3
):
    """
    This test is meant to test that the pyinstrument analysis retrieves the correct step's stats based on the specified
    interval. Stats are either retrieved from s3 or generated manually through python profiling.
    """
    if s3:
        # Fetch stats from s3
        os.makedirs(pyinstrument_dir)
        python_profile_analysis = PyinstrumentAnalysis(
            local_profile_dir=pyinstrument_dir, s3_path=bucket_prefix
        )
    else:
        # Do analysis and use those stats.
        _analysis_set_up(pyinstrument_python_profiler)
        python_profile_analysis = PyinstrumentAnalysis(local_profile_dir=pyinstrument_dir)
        _, bucket, prefix = is_s3(bucket_prefix)
        key = os.path.join(prefix, "framework", test_framework, PYINSTRUMENT_NAME)
        _upload_s3_folder(bucket, key, pyinstrument_dir)

    assert len(python_profile_analysis.python_profile_stats) == 3

    # Test that start_end_step_function call is recorded in received stats, but not end_start_step_function or
    # time_function.
    stats = python_profile_analysis.fetch_profile_stats_by_step(1)
    assert len(stats) == 1
    children = stats[0]["root_frame"]["children"]
    assert len(children) == 1 and children[0]["function"] == "start_end_step_function"

    # Test that end_start_step_function call is recorded in received stats, but not start_end_step_function or
    # time_function.
    stats = python_profile_analysis.fetch_profile_stats_by_step(
        1, end_step=2, start_phase=StepPhase.STEP_END, end_phase=StepPhase.STEP_START
    )
    assert len(stats) == 1
    children = stats[0]["root_frame"]["children"]
    assert len(children) == 1 and children[0]["function"] == "end_start_step_function"

    # Test that time_function call is recorded in received stats, but not step_function
    time_function_step_stats = python_profile_analysis.python_profile_stats[-1]
    step_start_time = (
        time_function_step_stats.start_time_since_epoch_in_micros / CONVERT_TO_MICROSECS
    )
    stats = python_profile_analysis.fetch_profile_stats_by_time(step_start_time, time.time())
    assert len(stats) == 1
    children = stats[0]["root_frame"]["children"]
    assert len(children) == 1 and children[0]["function"] == "time_function"
