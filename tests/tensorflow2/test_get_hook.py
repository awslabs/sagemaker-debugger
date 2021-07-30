# Standard Library
from statistics import mean

# Third Party
import pytest
from tensorflow.python.util.smdebug import get_smdebug_hook
from tests.utils import SagemakerSimulator, Timer, is_running_in_codebuild


@pytest.mark.skipif(
    is_running_in_codebuild() is False,
    reason="Microbenchmarking thresholds have been determined only for ci",
)
@pytest.mark.parametrize("use_smdebug", ["0", "1"])
def test_get_smdebug_hook_use_smdebug(
    use_smdebug, microbenchmark_repeat, microbenchmark_range, monkeypatch
):
    monkeypatch.setenv("USE_SMDEBUG", use_smdebug)
    times_taken = []
    for _ in range(microbenchmark_repeat):
        timer_context = Timer()
        with timer_context as t:
            for _ in range(microbenchmark_range):
                hook = get_smdebug_hook("keras")
        times_taken.append(t.time_taken)

    mean_time_taken = mean(times_taken)
    # time taken with use_smdebug == 0 is 3 seconds
    # time taken with use_smdebug == 1 is 21 seconds
    threshold = 3 if use_smdebug == "0" else 21
    assert mean_time_taken < threshold


@pytest.mark.skipif(
    is_running_in_codebuild() is False,
    reason="Microbenchmarking thresholds have been determined only for ci",
)
def test_sagemaker_context(microbenchmark_repeat, microbenchmark_range):
    times_taken = []
    for _ in range(microbenchmark_repeat):
        timer_context = Timer()
        with SagemakerSimulator():
            with timer_context as t:
                for _ in range(microbenchmark_range):
                    hook = get_smdebug_hook("keras")
            times_taken.append(t.time_taken)

    mean_time_taken = mean(times_taken)
    assert mean_time_taken < 12  # current mean = ~12 seconds
