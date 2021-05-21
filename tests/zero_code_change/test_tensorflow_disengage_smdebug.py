# Third Party
# Standard Library
import os
from unittest.mock import patch

import pytest
from tests.zero_code_change.test_tensorflow2_integration import helper_keras_fit

# First Party
import smdebug.tensorflow as smd
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
def test_tensorflow2_with_unsupported_version(eager_mode: bool = True):
    """ Test the default ZCC behavior of saving losses and metrics in eager and non-eager modes."""
    smd.del_hook()
    helper_keras_fit()
    hook = smd.get_hook()
    assert hook is None
