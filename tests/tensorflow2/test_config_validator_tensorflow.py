# Future
from __future__ import print_function

# Standard Library
import os
from unittest.mock import patch

# Third Party
import pytest

# First Party
from smdebug.core.config_validator import reset_config_validator
from smdebug.profiler.profiler_config_parser import reset_profiler_config_parser
from smdebug.tensorflow.singleton_utils import del_hook, get_hook


@pytest.fixture(autouse=True)
def cleanup():
    del_hook()
    yield
    os.environ.pop("USE_SMDEBUG", None)
    os.environ.pop("SM_HPS", None)
    reset_config_validator()
    reset_profiler_config_parser()


@patch("smdebug.core.config_validator.is_framework_version_supported", return_value=False)
def test_supported_tensorflow_version(is_framework_version_supported):
    del_hook()
    hook = get_hook()
    assert hook is None
