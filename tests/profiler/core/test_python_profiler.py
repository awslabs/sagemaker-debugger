# Standard Library
import json
import os
import pstats
import shutil
import time

# Third Party
import pytest

# First Party
from smdebug.profiler.analysis.python_profile_analysis import PyinstrumentAnalysis, cProfileAnalysis
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS, PYTHON_STATS_FILENAME
from smdebug.profiler.python_profiler import PyinstrumentPythonProfiler, cProfilePythonProfiler


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
    return "{0}/framework/{1}/{2}".format(out_dir, test_framework, cProfilePythonProfiler.name)


@pytest.fixture
def pyinstrument_dir(out_dir, test_framework):
    return "{0}/framework/{1}/{2}".format(out_dir, test_framework, PyinstrumentPythonProfiler.name)


@pytest.fixture(autouse=True)
def reset_python_profiler_dir(cprofile_dir, pyinstrument_dir):
    shutil.rmtree(cprofile_dir, ignore_errors=True)
    shutil.rmtree(pyinstrument_dir, ignore_errors=True)


@pytest.fixture()
def bucket_prefix():
    return "s3://smdebug-testing/resources/python_profile"


@pytest.fixture(scope="session")
def sample_times():
    return [time.time() for _ in range(5)]


def _profiling_set_up(python_profiler, start_step, end_step):
    current_step = start_step

    while current_step < end_step:
        python_profiler.start_profiling(current_step)
        assert python_profiler._step == current_step
        current_step += 1
        python_profiler.stop_profiling()


def _analysis_set_up(python_profiler):
    def step_function():
        time.sleep(
            0.0011
        )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds

    def time_function():
        time.sleep(
            0.0011
        )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds

    # step_function will be recorded in first step, time_function in the second.
    python_profiler.start_profiling(1)
    step_function()
    python_profiler.stop_profiling()
    python_profiler.start_profiling(2)
    time_function()
    python_profiler.stop_profiling()


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
        full_stats_path = os.path.join(cprofile_dir, stats_dir, PYTHON_STATS_FILENAME)
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
        full_stats_path = os.path.join(pyinstrument_dir, stats_dir, PYTHON_STATS_FILENAME + ".json")
        assert os.path.isfile(full_stats_path)
        with open(full_stats_path, "r") as f:
            assert json.load(f)  # validate output file


@pytest.mark.parametrize("s3", [True, False])
def test_cprofile_analysis(cprofile_python_profiler, cprofile_dir, bucket_prefix, s3):
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

    # Test that step_function call is recorded in received stats, but not time_function.
    assert len(python_profile_analysis.python_profile_stats) == 2

    stats = python_profile_analysis.fetch_profile_stats_by_step(1, 2)
    function_stats_list = stats.function_stats_list
    assert len(function_stats_list) > 0
    assert any(["step_function" in stat.function_name for stat in function_stats_list])
    assert all(["time_function" not in stat.function_name for stat in function_stats_list])

    # Test that time_function call is recorded in received stats, but not step_function
    time_function_step_stats = python_profile_analysis.python_profile_stats[-1]
    step_start_time = (
        time_function_step_stats.start_time_since_epoch_in_micros / CONVERT_TO_MICROSECS
    )
    stats = python_profile_analysis.fetch_profile_stats_by_time(step_start_time, time.time())
    function_stats_list = stats.function_stats_list
    assert len(function_stats_list) > 0
    assert any(["time_function" in stat.function_name for stat in function_stats_list])
    assert all(["step_function" not in stat.function_name for stat in function_stats_list])


@pytest.mark.parametrize("s3", [True, False])
def test_pyinstrument_analysis(pyinstrument_python_profiler, pyinstrument_dir, bucket_prefix, s3):
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

    # Test that step_function call is recorded in received stats, but not time_function.
    assert len(python_profile_analysis.python_profile_stats) == 2

    stats = python_profile_analysis.fetch_profile_stats_by_step(1, 2)
    assert len(stats) == 1
    children = stats[0]["root_frame"]["children"]
    assert len(children) == 1 and children[0]["function"] == "step_function"

    # Test that time_function call is recorded in received stats, but not step_function
    time_function_step_stats = python_profile_analysis.python_profile_stats[-1]
    step_start_time = (
        time_function_step_stats.start_time_since_epoch_in_micros / CONVERT_TO_MICROSECS
    )
    stats = python_profile_analysis.fetch_profile_stats_by_time(step_start_time, time.time())
    assert len(stats) == 1
    children = stats[0]["root_frame"]["children"]
    assert len(children) == 1 and children[0]["function"] == "time_function"
