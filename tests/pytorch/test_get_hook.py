# Third Party
import pytest
import torch
from numpy import percentile
from tests.utils import SagemakerSimulator, Timer, is_running_in_codebuild

# First Party
from smdebug.core.utils import FRAMEWORK, is_framework_version_supported


def test_did_you_forget_to_update_the_supported_framework_version():
    """
    This test is designed to save you 2 days of debugging time in case you're new
    to the codebase.

    One of the side-effects of not updating SUPPORTED_PT_VERSION_THRESHOLD with the
    version of Tensorflow you're trying to release for is that `smdebug.tensorflow.get_hook()` is going to return `None`
    and many tests will fail.

    This is just to make you aware of the problem.
    """
    if not is_framework_version_supported(FRAMEWORK.PYTORCH):
        raise Exception(
            "You are running against an unsupported version of Pytorch... apparently."
            " Please update `smdebug.pytorch.utils.SUPPORTED_TF_VERSION_THRESHOLD`"
            f" if you are trying to release sagemaker-debugger for the next version of pytorch ({torch.__version__})."
        )


@pytest.mark.skipif(
    is_running_in_codebuild() is False,
    reason="Microbenchmarking thresholds have been determined only for ci",
)
@pytest.mark.parametrize("use_smdebug", ["0", "1"])
def test_get_smdebug_hook_use_smdebug(
    use_smdebug, microbenchmark_repeat, microbenchmark_range, monkeypatch
):
    try:
        from torch.utils.smdebug import get_smdebug_hook
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
    except ImportError:
        print("Test needs framework hooks")


@pytest.mark.skipif(
    is_running_in_codebuild() is False,
    reason="Microbenchmarking thresholds have been determined only for ci",
)
def test_sagemaker_context(microbenchmark_repeat, microbenchmark_range):
    try:
        from torch.utils.smdebug import get_smdebug_hook
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
    except ImportError:
        print("Test needs framework hooks")
