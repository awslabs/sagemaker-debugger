# Standard Library
import os
import shutil
import time

# Third Party
import pytest

# First Party
from smdebug.profiler.analysis.python_profile_analysis import PythonProfileAnalysis
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS, PYTHON_STATS_FILENAME
from smdebug.profiler.python_profiler import PythonProfiler


@pytest.fixture
def test_framework():
    return "test-framework"


@pytest.fixture
def python_profile_folder(out_dir, test_framework):
    return "{0}/framework/{1}/python_profile".format(out_dir, test_framework)


@pytest.fixture(autouse=True)
def reset_python_profiler_folder(python_profile_folder):
    shutil.rmtree(python_profile_folder, ignore_errors=True)


@pytest.fixture
def python_profiler(out_dir, test_framework):
    return PythonProfiler(out_dir, test_framework)


@pytest.fixture(scope="session")
def sample_times():
    return [time.time() for _ in range(5)]


@pytest.mark.parametrize("steps", [(1, 2), (1, 5)])
def test_profiling(python_profiler, python_profile_folder, steps):
    start_step, end_step = steps
    current_step = start_step

    while current_step < end_step:
        python_profiler.start_profiling(current_step)
        assert python_profiler._step == current_step
        current_step += 1
        python_profiler.stop_profiling()

    # Test folder and corresponding files exist.
    assert os.path.isdir(python_profile_folder)

    stats_folders = os.listdir(python_profile_folder)
    assert len(stats_folders) == (end_step - start_step)
    for stats_folder in stats_folders:
        full_stats_path = os.path.join(python_profile_folder, stats_folder, PYTHON_STATS_FILENAME)
        assert os.path.isfile(full_stats_path)


def test_analysis(python_profiler, python_profile_folder):
    def step_function():
        pass

    def time_function():
        pass

    # step_function will be recorded in first step, time_function in the second.
    python_profiler.start_profiling(1)
    step_function()
    python_profiler.stop_profiling()
    python_profiler.start_profiling(2)
    time_function()
    python_profiler.stop_profiling()

    python_profile_analysis = PythonProfileAnalysis(python_profile_folder)

    # Test that step_function call is recorded in received stats, but not time_function.
    assert len(python_profile_analysis.records) == 2

    stats = python_profile_analysis.fetch_python_profile_stats_by_step(1, 2)
    assert len(stats) > 0
    assert any(["step_function" in stat.function_name for stat in stats])
    assert all(["time_function" not in stat.function_name for stat in stats])

    # Test that time_function call is recorded in received stats, but not step_function
    time_function_record = python_profile_analysis.records[-1]
    record_start_time = time_function_record.start_time_since_epoch_in_micros / CONVERT_TO_MICROSECS
    stats = python_profile_analysis.fetch_python_profile_stats_by_time(
        record_start_time, time.time()
    )
    assert len(stats) > 0
    assert any(["time_function" in stat.function_name for stat in stats])
    assert all(["step_function" not in stat.function_name for stat in stats])
