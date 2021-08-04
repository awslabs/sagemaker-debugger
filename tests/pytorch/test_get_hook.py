# Standard Library
# Third Party
import pytest
from numpy import percentile
from tests.utils import SagemakerSimulator, Timer, is_running_in_codebuild
from torch.utils.smdebug import get_smdebug_hook


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
                hook = get_smdebug_hook()
        times_taken.append(t.time_taken)

    p95 = percentile(times_taken, 95)
    # mean time taken with use_smdebug == 0 is 3 seconds
    # mean time taken with use_smdebug == 1 is 10 seconds
    threshold = 3 if use_smdebug == "0" else 10
    assert p95 < threshold


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
                    hook = get_smdebug_hook()
            times_taken.append(t.time_taken)

    p95 = percentile(times_taken, 95)
    assert p95 < 10  # current mean = ~10 seconds
