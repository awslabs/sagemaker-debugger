# Third Party
import torch

# First Party
from smdebug.core.utils import FRAMEWORK, is_framework_version_supported


def test_did_you_forget_to_update_the_supported_framework_version():
    """
    This test is designed to save you 2 days of debugging time in case you're new
    to the codebase.

    One of the side-effects of not updating LATEST_SUPPORTED_TF_VERSION with the
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
