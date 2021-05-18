# Future
from __future__ import print_function

# Standard Library
import os
from unittest.mock import patch

# Third Party
import pytest

# First Party
import smdebug.pytorch.singleton_utils
from smdebug.core.config_validator import reset_config_validator


@pytest.fixture(autouse=True)
def cleanup():
    smdebug.pytorch.singleton_utils.del_hook()
    yield
    os.environ.pop("USE_SMDEBUG", None)
    os.environ.pop("SM_HPS", None)
    reset_config_validator()


@patch("smdebug.core.config_validator.is_framework_version_supported", return_value=False)
def test_supported_pytorch_version(is_framework_version_supported):
    smdebug.pytorch.singleton_utils.del_hook()
    hook = smdebug.pytorch.singleton_utils.get_hook()
    assert hook is None
