# Standard Library
import time

# First Party
from smdebug.profiler.profiler_constants import (
    PROFILER_DURATION_DEFAULT,
    PROFILER_NUM_STEPS_DEFAULT,
)

current_step = 3
current_time = time.time()
good_start_step = 3
bad_start_step = 1
num_steps = 2
good_start_time = current_time
bad_start_time = current_time - 1000
duration = 500

# Each test case consists of (detailed_profiling_parameters, expected_detailed_profiling_enabled, expected_can_profile expected_values) where:
#   - detailed_profiling_parameters refers to the fields (if they exist, `None` otherwise) in the detailed profiling config,
#       i.e. (start_step, num_steps, start_time, duration)
#   - expected_detailed_profiling_enabled refers to the expected value of detailed_profiling_enabled
#   - expected_can_detailed_profile refers to the expected value of can_start_detailed_profiling()
#   - expected_values refers to expected values of the profile range after parsing, i.e.
#       (start_step, end_step, start_time, end_time)
detailed_profiling_test_cases = [
    (
        # valid case where both start_step and num_steps are provided. profiler starts at start_step and profiles for num_steps steps.
        (good_start_step, num_steps, None, None),
        True,
        True,
        (good_start_step, good_start_step + num_steps, None, None),
    ),
    (
        # valid case where only start_step is provided. profiler starts at start_step and profiles for PROFILER_NUM_STEPS_DEFAULT steps.
        (good_start_step, None, None, None),
        True,
        True,
        (good_start_step, good_start_step + PROFILER_NUM_STEPS_DEFAULT, None, None),
    ),
    # valid case where only num_steps is provided. profiler starts at current_step and profiles for num_steps steps.
    (
        (None, num_steps, None, None),
        True,
        True,
        (current_step, current_step + num_steps, None, None),
    ),
    # valid case where start_time and duration are provided. profiler starts at start_time and profiles for duration seconds.
    (
        (None, None, good_start_time, duration),
        True,
        True,
        (None, None, good_start_time, good_start_time + duration),
    ),
    (
        # valid case where only start_time is provided. profiler starts at start_time and profiles until the next step.
        (None, None, good_start_time, None),
        True,
        True,
        (None, current_step + 1, good_start_time, None),
    ),
    # valid case where only duration is provided. profiler starts immediately and profiles for duration seconds.
    ((None, None, None, duration), True, True, (None, None, current_time, current_time + duration)),
    (
        # valid case where detailed_profiling_enabled is True, but start_step is too small.
        # since bad_start_step + PROFILER_NUM_STEPS_DEFAULT < current_step, no profiling takes place.
        (bad_start_step, None, None, None),
        True,
        False,
        (bad_start_step, bad_start_step + PROFILER_NUM_STEPS_DEFAULT, None, None),
    ),
    # valid case where detailed_profiling_enabled is True, but start_time is too small.
    # since start_time + duration are is before the current time, no profiling takes place.
    (
        (None, None, bad_start_time, duration),
        True,
        False,
        (None, None, bad_start_time, bad_start_time + duration),
    ),
    # invalid case where both step and time fields are provided, which is not allowed. No profiling takes place.
    (
        (good_start_step, num_steps, good_start_time, duration),
        False,
        False,
        (good_start_step, None, good_start_time, None),
    ),
    # invalid case where no fields are provided. No profiling takes place.
    ((None, None, None, None), False, False, (None, None, None, None)),
]
