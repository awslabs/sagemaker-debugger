# Standard Library
import time

# First Party
from smdebug.profiler.profiler_constants import (
    PROFILER_DURATION_DEFAULT,
    PROFILER_NUM_STEPS_DEFAULT,
)

current_step = 3
good_start_step = 3
bad_start_step = 1
num_steps = 2
good_start_time = time.time()
bad_start_time = time.time() - 1000
duration = 500

# Each test case consists of (detailed_profiling_parameters, expected_values) where:
#   - detailed_profiling_parameters refers to the fields (if they exist, `None` otherwise) in the detailed profiling config,
#       i.e. (start_step, num_steps, start_time, duration)
#   - expected_values refers to expected values of the profile range after parsing, i.e.
#       (profiler_start, profiler_end, can_enable_profiling())
detailed_profiling_test_cases = [
    (
        # valid case where both start_step and num_steps are provided. profiler starts at start_step and profiles for num_steps steps.
        (good_start_step, num_steps, None, None),
        (good_start_step, good_start_step + num_steps, True),
    ),
    (
        # valid case where only start_step is provided. profiler starts at start_step and profiles for PROFILER_NUM_STEPS_DEFAULT steps.
        (good_start_step, None, None, None),
        (good_start_step, good_start_step + PROFILER_NUM_STEPS_DEFAULT, True),
    ),
    # valid case where only num_steps is provided. profiler starts at current_step and profiles for num_steps steps.
    ((None, num_steps, None, None), (current_step, current_step + num_steps, True)),
    # valid case where start_time and duration are provided. profiler starts at start_time and profiles for duration seconds.
    ((None, None, good_start_time, duration), (good_start_time, good_start_time + duration, True)),
    (
        # valid case where only start_time is provided. profiler starts at start_time and profiles until the next step.
        (None, None, good_start_time, None),
        (good_start_time, good_start_time + PROFILER_DURATION_DEFAULT, True),
    ),
    # valid case where only duration is provided. profiler starts immediately and profiles for duration seconds.
    ((None, None, None, duration), (None, None, True)),
    (
        # invalid case where start_step is too small. since bad_start_step + PROFILER_NUM_STEPS_DEFAULT < current_step, no profiling takes place.
        (bad_start_step, None, None, None),
        (bad_start_step, bad_start_step + PROFILER_NUM_STEPS_DEFAULT, False),
    ),
    # invalid case where start_time is too small. since start_time + duration are is before the current time, no profiling takes place.
    ((None, None, bad_start_time, duration), (bad_start_time, bad_start_time + duration, False)),
    # invalid case where both step and time fields are provided, which is not allowed. No profiling takes place.
    ((good_start_step, num_steps, good_start_time, duration), (None, None, False)),
    # invalid case where no fields are provided. No profiling takes place.
    ((None, None, None, None), (None, None, False)),
]
