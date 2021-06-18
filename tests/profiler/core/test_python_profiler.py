# Standard Library
import os
import shutil
import time
from multiprocessing.pool import ThreadPool

# Third Party
import boto3
import pandas as pd
import pytest
from tests.profiler.core.utils import validate_python_profiling_stats

# First Party
from smdebug.core.access_layer.utils import is_s3
from smdebug.core.utils import FRAMEWORK
from smdebug.profiler.analysis.python_profile_analysis import PyinstrumentAnalysis, cProfileAnalysis
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profile_utils import PythonProfileModes, StepPhase
from smdebug.profiler.python_profiler import (
    PyinstrumentPythonProfiler,
    cProfilePythonProfiler,
    cProfileTimer,
)


@pytest.fixture
def framework():
    return FRAMEWORK.TENSORFLOW


@pytest.fixture()
def cprofile_python_profiler(out_dir, framework):
    return cProfilePythonProfiler(out_dir, framework, cProfileTimer.TOTAL_TIME)


@pytest.fixture()
def pyinstrument_python_profiler(out_dir, framework):
    return PyinstrumentPythonProfiler(out_dir, framework)


@pytest.fixture()
def framework_dir(out_dir, framework):
    return "{0}/framework/{1}".format(out_dir, framework.value)


@pytest.fixture(autouse=True)
def reset_python_profiler_dir(framework_dir):
    shutil.rmtree(framework_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def bucket_prefix():
    return f"s3://smdebug-testing/resources/python_profile/{int(time.time())}"


def pre_step_zero_function():
    time.sleep(
        0.0011
    )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds


def start_end_step_function():
    time.sleep(
        0.0011
    )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds


def end_start_step_function():
    time.sleep(
        0.0011
    )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds


def between_modes_function():
    time.sleep(
        0.0011
    )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds


def eval_function():
    time.sleep(
        0.0011
    )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds


def post_hook_close_function():
    time.sleep(
        0.0011
    )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds


def time_function():
    time.sleep(
        0.0011
    )  # stall long enough to be recorded by pyinstrument, which records every 0.001 seconds


def _upload_s3_folder(bucket, key, folder):
    s3_client = boto3.client("s3")
    filenames = []
    for root, _, files in os.walk(folder):
        for file in files:
            node_id = os.path.basename(os.path.dirname(root))
            stats_dir = os.path.basename(root)
            full_key = os.path.join(key, node_id, stats_dir, file)
            filenames.append((os.path.join(root, file), bucket, full_key))

    def upload_files(args):
        s3_client.upload_file(*args)

    pool = ThreadPool(processes=10)
    pool.map(upload_files, filenames)


def _validate_analysis(profiler_name, stats, expected_functions):
    function_names = [
        pre_step_zero_function.__name__,
        start_end_step_function.__name__,
        end_start_step_function.__name__,
        between_modes_function.__name__,
        eval_function.__name__,
        post_hook_close_function.__name__,
        time_function.__name__,
    ]

    assert stats is not None, "No stats found!"

    for analysis_function in function_names:
        if profiler_name == CPROFILE_NAME:
            function_stats_list = stats.function_stats_list
            assert len(function_stats_list) > 0

            if analysis_function in expected_functions:
                assert any(
                    [analysis_function in stat.function_name for stat in function_stats_list]
                ), f"{analysis_function} should be found in function stats!"
            else:
                assert all(
                    [analysis_function not in stat.function_name for stat in function_stats_list]
                ), f"{analysis_function} should not be found in function stats!"
        else:
            assert len(stats) == 1
            actual_functions = map(
                lambda x: x["function"], stats[0].json_stats["root_frame"]["children"]
            )
            assert set(actual_functions) == set(expected_functions)


@pytest.mark.parametrize("use_pyinstrument", [False, True])
@pytest.mark.parametrize("steps", [(1, 2), (1, 5)])
def test_python_profiling(
    use_pyinstrument, cprofile_python_profiler, pyinstrument_python_profiler, framework_dir, steps
):
    if use_pyinstrument:
        python_profiler = pyinstrument_python_profiler
        profiler_name = PYINSTRUMENT_NAME
    else:
        python_profiler = cprofile_python_profiler
        profiler_name = CPROFILE_NAME

    python_stats_dir = os.path.join(framework_dir, profiler_name)

    start_step, end_step = steps
    current_step = start_step

    while current_step < end_step:
        python_profiler.start_profiling(StepPhase.STEP_START, start_step=current_step)
        assert python_profiler._start_step == current_step
        assert python_profiler._start_phase == StepPhase.STEP_START
        python_profiler.stop_profiling(StepPhase.STEP_END, current_step)
        current_step += 1

    expected_stats_dir_count = end_step - start_step
    validate_python_profiling_stats(python_stats_dir, profiler_name, expected_stats_dir_count)


@pytest.mark.parametrize("use_pyinstrument", [False, True])
@pytest.mark.parametrize("s3", [False, True])
def test_python_analysis(
    use_pyinstrument,
    cprofile_python_profiler,
    pyinstrument_python_profiler,
    framework_dir,
    framework,
    bucket_prefix,
    s3,
):
    """
    This test is meant to test that the cProfile/pyinstrument analysis retrieves the correct step's stats based on the
    specified interval. Stats are either retrieved from s3 or generated manually through python profiling.
    """
    if use_pyinstrument:
        python_profiler = pyinstrument_python_profiler
        analysis_class = PyinstrumentAnalysis
        profiler_name = PYINSTRUMENT_NAME
        num_expected_files = 14
    else:
        python_profiler = cprofile_python_profiler
        analysis_class = cProfileAnalysis
        profiler_name = CPROFILE_NAME
        num_expected_files = 7

    python_stats_dir = os.path.join(framework_dir, profiler_name)

    if s3:
        # Fetch stats from s3
        os.makedirs(python_stats_dir)
        python_profile_analysis = analysis_class(
            local_profile_dir=python_stats_dir, s3_path=bucket_prefix
        )
    else:
        # Do analysis and use those stats.

        # pre_step_zero_function is called in between the start of the script and the start of first step of TRAIN.
        python_profiler.start_profiling(StepPhase.START)
        pre_step_zero_function()
        python_profiler.stop_profiling(
            StepPhase.STEP_START, end_mode=PythonProfileModes.TRAIN, end_step=1
        )

        # start_end_step_function is called in between the start and end of first step of TRAIN.
        python_profiler.start_profiling(
            StepPhase.STEP_START, start_mode=PythonProfileModes.TRAIN, start_step=1
        )
        start_end_step_function()
        python_profiler.stop_profiling(
            StepPhase.STEP_END, end_mode=PythonProfileModes.TRAIN, end_step=1
        )

        # end_start_step_function is called in between the end of first step and the start of second step of TRAIN.
        python_profiler.start_profiling(
            StepPhase.STEP_END, start_mode=PythonProfileModes.TRAIN, start_step=1
        )
        end_start_step_function()
        python_profiler.stop_profiling(
            StepPhase.STEP_START, end_mode=PythonProfileModes.TRAIN, end_step=2
        )

        # train_and_eval function is called in between the TRAIN and EVAL modes.
        python_profiler.start_profiling(
            StepPhase.STEP_END, start_mode=PythonProfileModes.TRAIN, start_step=1
        )
        between_modes_function()
        python_profiler.stop_profiling(
            StepPhase.STEP_START, end_mode=PythonProfileModes.EVAL, end_step=1
        )

        # eval function is called in between the start and end of first step of EVAL.
        python_profiler.start_profiling(
            StepPhase.STEP_START, start_mode=PythonProfileModes.EVAL, start_step=1
        )
        eval_function()
        python_profiler.stop_profiling(
            StepPhase.STEP_END, end_mode=PythonProfileModes.EVAL, end_step=1
        )

        # post_hook_close_function is called in between the end of the last step of EVAL and the end of the script.
        python_profiler.start_profiling(
            StepPhase.STEP_END, start_mode=PythonProfileModes.EVAL, start_step=1
        )
        post_hook_close_function()
        python_profiler.stop_profiling(StepPhase.END)

        # time function is called in between start and end of second step of TRAIN.
        # NOTE: This needs to be profiled last for tests to pass.
        python_profiler.start_profiling(
            StepPhase.STEP_START, start_mode=PythonProfileModes.TRAIN, start_step=2
        )
        time_function()
        python_profiler.stop_profiling(
            StepPhase.STEP_END, end_mode=PythonProfileModes.TRAIN, end_step=2
        )

        python_profile_analysis = analysis_class(local_profile_dir=python_stats_dir)
        _, bucket, prefix = is_s3(bucket_prefix)
        key = os.path.join(prefix, "framework", framework.value, profiler_name)
        _upload_s3_folder(bucket, key, python_stats_dir)

    python_profile_stats_df = python_profile_analysis.list_profile_stats()
    assert isinstance(python_profile_stats_df, pd.DataFrame)
    assert python_profile_stats_df.shape[0] == num_expected_files

    # Test that pre_step_zero_function call is recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_pre_step_zero_profile_stats(refresh_stats=False)
    _validate_analysis(profiler_name, stats, [pre_step_zero_function.__name__])

    # Test that start_end_step_function call is recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_profile_stats_by_step(1, refresh_stats=False)
    _validate_analysis(profiler_name, stats, [start_end_step_function.__name__])

    # Test that end_start_step_function call is recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_profile_stats_by_step(
        1,
        end_step=2,
        start_phase=StepPhase.STEP_END,
        end_phase=StepPhase.STEP_START,
        refresh_stats=False,
    )
    _validate_analysis(profiler_name, stats, [end_start_step_function.__name__])

    # Test that train_and_eval_function call is recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_profile_stats_between_modes(
        PythonProfileModes.TRAIN, PythonProfileModes.EVAL, refresh_stats=False
    )
    _validate_analysis(profiler_name, stats, [between_modes_function.__name__])

    # Test that eval_function call is recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_profile_stats_by_step(
        1, mode=PythonProfileModes.EVAL, refresh_stats=False
    )
    _validate_analysis(profiler_name, stats, [eval_function.__name__])

    # Test that pre_step_zero_function call is recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_post_hook_close_profile_stats(refresh_stats=False)
    _validate_analysis(profiler_name, stats, [post_hook_close_function.__name__])

    # Test that time_function call is recorded in received stats, but not the other functions.
    time_function_step_stats = python_profile_analysis.python_profile_stats[-1]
    step_start_time = (
        time_function_step_stats.start_time_since_epoch_in_micros / CONVERT_TO_MICROSECS
    )
    stats = python_profile_analysis.fetch_profile_stats_by_time(
        step_start_time, time.time(), refresh_stats=False
    )
    _validate_analysis(profiler_name, stats, [time_function.__name__])

    # Following analysis functions are for cProfile only
    if use_pyinstrument:
        return

    # Test that functions called in TRAIN are recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_profile_stats_by_training_phase(refresh_stats=False)[
        (PythonProfileModes.TRAIN, PythonProfileModes.TRAIN)
    ]
    _validate_analysis(
        profiler_name,
        stats,
        [
            start_end_step_function.__name__,
            end_start_step_function.__name__,
            time_function.__name__,
        ],
    )

    # Test that functions called in training loop are recorded in received stats, but not the other functions.
    stats = python_profile_analysis.fetch_profile_stats_by_job_phase(refresh_stats=False)[
        "training_loop"
    ]
    _validate_analysis(
        profiler_name,
        stats,
        [
            start_end_step_function.__name__,
            end_start_step_function.__name__,
            between_modes_function.__name__,
            eval_function.__name__,
            time_function.__name__,
        ],
    )
