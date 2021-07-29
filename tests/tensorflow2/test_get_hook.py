# Standard Library
import os
from statistics import mean

# Third Party
import pytest
from tensorflow.python.util.smdebug import get_smdebug_hook
from tests.utils import SagemakerSimulator, Timer


@pytest.mark.parametrize("use_smdebug", ["0", "1"])
def test_get_smdebug_hook_use_smdebug(
    use_smdebug, microbenchmark_repeat_constant, microbenchmark_range_constant, monkeypatch
):
    monkeypatch.setenv("USE_SMDEBUG", use_smdebug)
    times_taken = []
    for _ in range(microbenchmark_repeat_constant):
        timer_context = Timer()
        with timer_context as t:
            for _ in range(microbenchmark_range_constant):
                hook = get_smdebug_hook("keras")
        times_taken.append(t.time_taken)

    mean_time_taken = mean(times_taken)
    if use_smdebug == "0":
        assert mean_time_taken < 2, os.getenv(
            "$CODEBUILD_SRC_DIR", "No Value"
        )  # current mean = ~1.8 seconds
    else:
        assert mean_time_taken < 15, os.getenv(
            "$CODEBUILD_SRC_DIR", "No Value"
        )  # current mean = ~14.5 seconds


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
    assert mean_time_taken < 7, os.getenv(
        "$CODEBUILD_SRC_DIR", "No Value"
    )  # current mean = ~6.9 seconds
