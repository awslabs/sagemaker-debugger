# Standard Library
import re
import time

# First Party
from smdebug.profiler.profiler_constants import (
    CPROFILE_NAME,
    PROFILING_NUM_STEPS_DEFAULT,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profiler import cProfileTimer

current_step = 3
current_time = time.time()
good_start_step = 3
bad_start_step = 1
bad_start_step_2 = 5
num_steps = 2
good_start_time = current_time
bad_start_time = current_time - 1000
duration = 500


# These test cases will primarily test the various combinations of start step, num steps, start_time, duration for
# detailed profiling. Each test case consists of (detailed_profiling_parameters, expected_enabled,
# expected_can_profile, expected_values) where:
#   - detailed_profiling_parameters refers to fields (if they exist, `None` otherwise) in the detailed profiling config,
#       i.e. (start_step, num_steps, start_time, duration)
#   - expected_enabled refers to whether detailed profiling is enabled (no errors parsing config).
#   - expected_can_profile refers to the expected value of should_save_metrics for detailed profiling
#   - expected_values refers to expected values of the profile range after parsing, i.e.
#       (start_step, end_step, start_time, end_time)
detailed_profiling_test_cases = [
    # Valid case where both start_step and num_steps are provided. Profiler starts at start_step and profiles for
    # num_steps steps. Profiler will profile current step.
    (
        (good_start_step, num_steps, None, None),
        True,
        True,
        (good_start_step, good_start_step + num_steps, None, None),
    ),
    # Valid case where only start_step is provided. Profiler starts at start_step and profiles for
    # PROFILER_NUM_STEPS_DEFAULT steps. Profiler will profile current step.
    (
        (good_start_step, None, None, None),
        True,
        True,
        (good_start_step, good_start_step + PROFILING_NUM_STEPS_DEFAULT, None, None),
    ),
    # Valid case where only num_steps is provided. Profiler starts at current_step and profiles for num_steps steps.
    # Profiler will profile current step.
    (
        (None, num_steps, None, None),
        True,
        True,
        (current_step, current_step + num_steps, None, None),
    ),
    # Valid case where start_time and duration are provided. Profiler starts at start_time and profiles for duration
    # seconds. Profiler will profile current step.
    (
        (None, None, good_start_time, duration),
        True,
        True,
        (None, None, good_start_time, good_start_time + duration),
    ),
    # Valid case where only start_time is provided. Profiler starts at start_time and profiles until the next step.
    # Profiler will profile current step.
    (
        (None, None, good_start_time, None),
        True,
        True,
        (None, current_step + 1, good_start_time, None),
    ),
    # Valid case where only duration is provided. Profiler starts immediately and profiles for duration seconds.
    # Profiler will profile current step.
    ((None, None, None, duration), True, True, (None, None, current_time, current_time + duration)),
    # Valid case where detailed_profiling_enabled is True, but start_step is too small. Profiler starts at
    # bad_start_step and profiles for PROFILER_NUM_STEPS_DEFAULT steps. because
    # bad_start_step + PROFILER_NUM_STEPS_DEFAULT < current_step, Profiler does not profile current step.
    (
        (bad_start_step, None, None, None),
        True,
        False,
        (bad_start_step, bad_start_step + PROFILING_NUM_STEPS_DEFAULT, None, None),
    ),
    # Valid case where detailed_profiling_enabled is True, but start_time is too small. Profiler starts at start time
    # and profiles for duration seconds. because bad_start_time + duration is before the current time, Profiler does
    # not profile current step.
    (
        (None, None, bad_start_time, duration),
        True,
        False,
        (None, None, bad_start_time, bad_start_time + duration),
    ),
    # Invalid case where both step and time fields are provided, which is not allowed. No detailed profiling takes
    # place.
    (
        (good_start_step, num_steps, good_start_time, duration),
        False,
        False,
        (good_start_step, None, good_start_time, None),
    ),
]

# These test cases will primarily test the various combinations of start step, metrics_regex and metrics_name for
# dataloader profiling. Each test case consists of (dataloader_parameters, expected_enabled, expected_can_profile,
# expected_values) where:
#   - dataloader_parameters refers to fields (if they exist, `None` otherwise) in the dataloader metrics config,
#       i.e. (start_step, metrics_regex, metrics__name)
#   - expected_enabled refers to whether dataloader metrics collection is enabled (no errors parsing config).
#   - expected_can_profile refers to the expected value should_save_metrics for dataloader
#   - expected_values refers to expected values of the profile range after parsing, i.e.
#       (start_step, end_step, metrics_regex)
dataloader_test_cases = [
    # Valid case where start step and metrics regex are provided. Metrics collection is done for the current step for
    # the given metrics name.
    (
        (good_start_step, "Dataloader:Event", "Dataloader:Event1"),
        True,
        True,
        (
            good_start_step,
            good_start_step + PROFILING_NUM_STEPS_DEFAULT,
            re.compile("dataloader:event"),
        ),
    ),
    # Valid case where start step and metrics regex are provided. Metrics collection is done for the current step, but
    # not for the given metrics name since the regex didn't match the name.
    (
        (good_start_step, "Dataloader:Event2", "Dataloader:Event1"),
        True,
        False,
        (good_start_step, None, re.compile("dataloader:event2")),
    ),
    # Valid case where start step is provided. Metrics collection is done for the current step for the given metrics
    # name.
    (
        (good_start_step, None, "Dataloader:Event1"),
        True,
        True,
        (good_start_step, good_start_step + PROFILING_NUM_STEPS_DEFAULT, re.compile(".*")),
    ),
    # Invalid case where start step and metrics regex are provided, but the metrics regex is invalid. No dataloader
    # metrics collection is done.
    ((good_start_step, "*", "Dataloader:Event1"), False, False, (None, None, None)),
]

# These test cases will primarily test the various combinations of start step, num steps, profiler name and cprofile
# timer for python profiling. Each test case consists of (python_profiling_parameters, expected_enabled,
# expected_can_profile, expected_values) where:
#   - python_profiling_parameters refers to fields (if they exist, `None` otherwise) in the python profiling config,
#       i.e. (start_step, num_steps, profiler_name, cprofile_timer)
#   - expected_enabled refers to whether python profiling is enabled (no errors parsing config).
#   - expected_can_profile refers to the expected value hould_save_metrics for python profiling
#   - expected_values refers to expected values of the profile range after parsing, i.e.
#       (start_step, end_step, profiler_name, cprofile_timer)
python_profiling_test_cases = [
    # Valid case where step fields, profiler name and cprofile timer are specified. Profiler starts at start step and
    # profiles for num_steps steps with cProfile measuring off cpu time. Profiler will profile current step.
    (
        (good_start_step, num_steps, CPROFILE_NAME, cProfileTimer.OFF_CPU_TIME.value),
        True,
        True,
        (good_start_step, good_start_step + num_steps, CPROFILE_NAME, cProfileTimer.OFF_CPU_TIME),
    ),
    # Valid case where only step fields are provided. Profiler starts at start_step and profiles for num_steps steps
    # with cProfile measuring total time. Profiler will profile current step.
    (
        (good_start_step, num_steps, None, None),
        True,
        True,
        (good_start_step, good_start_step + num_steps, CPROFILE_NAME, cProfileTimer.TOTAL_TIME),
    ),
    # Valid case where step fields and cprofile timer are provided. Profiler starts at start_step and profiles for
    # num_steps steps with cProfile measuring cpu time. Profiler will profile current step.
    (
        (good_start_step, num_steps, None, cProfileTimer.CPU_TIME.value),
        True,
        True,
        (good_start_step, good_start_step + num_steps, CPROFILE_NAME, cProfileTimer.CPU_TIME),
    ),
    # Valid case where step fields and profiler name are provided. Profiler starts at start_step and profiles for
    # num_steps steps with Pyinstrument. Profiler will profile current step.
    (
        (good_start_step, num_steps, PYINSTRUMENT_NAME, None),
        True,
        True,
        (good_start_step, good_start_step + num_steps, PYINSTRUMENT_NAME, None),
    ),
    # Valid case where step fields, profiler name and cprofile timer are provided. Profiler starts at start_step and
    # profiles for num_steps steps with Pyinstrument (since use pyinstrument is True, cprofile timer is ignored).
    # Profiler will profile current step.
    (
        (good_start_step, num_steps, PYINSTRUMENT_NAME, cProfileTimer.CPU_TIME.value),
        True,
        True,
        (good_start_step, good_start_step + num_steps, PYINSTRUMENT_NAME, None),
    ),
    # Invalid case where profiler name and cprofile timer are provided. No step or time range has been provided, so
    # profiler does not profile current step.
    (
        (None, None, CPROFILE_NAME, cProfileTimer.CPU_TIME.value),
        True,
        False,
        (None, None, CPROFILE_NAME, cProfileTimer.CPU_TIME),
    ),
    # Invalid case where step fields and profiler name are provided, but the profiler name is invalid. No python
    # profiling takes place.
    (
        (good_start_step, num_steps, "bad_profiler_name", None),
        False,
        False,
        (None, None, None, None),
    ),
    # Invalid case where step fields and cprofile timer are provided, but the cprofile timer is invalid. No python
    # profiling takes place.
    (
        (good_start_step, num_steps, CPROFILE_NAME, "bad_cprofile_timer"),
        False,
        False,
        (None, None, None, None),
    ),
]

# These test cases will primarily test the various combinations of start step, num steps that are unique to herring
# profiling. Each test case consists of (herring_profiling_parameters, expected_profiling_enabled,
# expected_can_profile, expected_values) where:
#   - smdataparallel_profiling_parameters refers to fields (if they exist, `None` otherwise) in the smdataparallel profiling config,
#       i.e. (start_step, num_steps)
#   - expected_profiling_enabled refers to whether herring profiling is enabled (no errors parsing config).
#   - expected_can_profile refers to the expected value of should_save_metrics for herring profiling
#   - expected_values refers to expected values of the profile range after parsing, i.e.
#       (start_step, end_step)
smdataparallel_profiling_test_cases = [
    # Valid case where both start_step and num_steps are provided. Profiler starts at start_step and profiles for
    # num_steps steps. Profiler will profile current step.
    ((good_start_step, num_steps), True, True, (good_start_step, good_start_step + num_steps)),
    # Valid case where only start_step is provided. Profiler starts at start_step and profiles for
    # PROFILER_NUM_STEPS_DEFAULT steps. Profiler will profile current step.
    (
        (good_start_step, None),
        True,
        True,
        (good_start_step, good_start_step + PROFILING_NUM_STEPS_DEFAULT),
    ),
    # Valid case where only num_steps is provided. Profiler starts at current_step and profiles for num_steps steps.
    # Profiler will profile current step.
    ((None, num_steps), True, True, (current_step, current_step + num_steps)),
    # Valid case where detailed_profiling_enabled is True, but start_step is too small. Profiler starts at
    # bad_start_step and profiles for PROFILER_NUM_STEPS_DEFAULT steps. because
    # bad_start_step + PROFILING_NUM_STEPS_DEFAULT < current_step, Profiler does not profile current step.
    (
        (bad_start_step_2, None),
        True,
        False,
        (bad_start_step_2, bad_start_step_2 + PROFILING_NUM_STEPS_DEFAULT),
    ),
]


def build_metrics_config(**config_parameters):
    return str({key: value for key, value in config_parameters.items() if value is not None})
