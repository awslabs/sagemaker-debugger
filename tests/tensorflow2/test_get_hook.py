# Standard Library
from statistics import mean

# Third Party
from tensorflow.python.util.smdebug import get_smdebug_hook
from tests.utils import SagemakerSimulator, Timer


def test_get_smdebug_hook_no_setup(microbenchmark_repeat_constant, microbenchmark_range_constant):
    times_taken = []
    for _ in range(microbenchmark_repeat_constant):
        timer_context = Timer()
        with timer_context as t:
            for _ in range(microbenchmark_range_constant):
                hook = get_smdebug_hook("keras")
        times_taken.append(t.time_taken)

    mean_time_taken = mean(times_taken)
    assert mean_time_taken < 15  # current mean = ~14.5 seconds


def test_get_smdebug_hook_use_smdebug(
    microbenchmark_repeat_constant, microbenchmark_range_constant, monkeypatch
):
    monkeypatch.setenv("USE_SMDEBUG", "0")
    times_taken = []
    for _ in range(microbenchmark_repeat_constant):
        timer_context = Timer()
        with timer_context as t:
            for _ in range(microbenchmark_range_constant):
                hook = get_smdebug_hook("keras")
        times_taken.append(t.time_taken)

    mean_time_taken = mean(times_taken)
    assert mean_time_taken < 2  # current mean = ~1.8 seconds


def test_sagemaker_context(microbenchmark_repeat_constant, microbenchmark_range_constant):
    times_taken = []
    for _ in range(microbenchmark_repeat_constant):
        timer_context = Timer()
        with SagemakerSimulator():
            with timer_context as t:
                for _ in range(microbenchmark_range_constant):
                    hook = get_smdebug_hook("keras")
            times_taken.append(t.time_taken)

    mean_time_taken = mean(times_taken)
    assert mean_time_taken < 7  # current mean = ~6.9 seconds
