# Third Party
# Standard Library
import os
from unittest.mock import patch

import pytest
from tests.zero_code_change.pt_utils import helper_torch_train

# First Party
import smdebug.pytorch as smd
from smdebug.core.config_validator import reset_config_validator
from smdebug.profiler.profiler_config_parser import reset_profiler_config_parser


@pytest.fixture(autouse=True)
def cleanup():
    yield
    os.environ.pop("USE_SMDEBUG", None)
    os.environ.pop("SM_HPS", None)
    reset_config_validator()
    reset_profiler_config_parser()


@patch("smdebug.core.config_validator.is_framework_version_supported", return_value=False)
def test_pytorch_with_unsupported_version(use_loss_module=False):
    smd.del_hook()
    helper_torch_train(script_mode=False, use_loss_module=use_loss_module)
    print("Finished Training")
    hook = smd.get_hook()
    assert hook is None
