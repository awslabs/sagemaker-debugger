# Future
from __future__ import print_function

# Standard Library
import os
from unittest.mock import patch

# Third Party
import pytest

# First Party
import smdebug.pytorch.singleton_utils
from smdebug.core.config_validator import get_config_validator, reset_config_validator


@pytest.fixture(autouse=True)
def cleanup():
    smdebug.pytorch.singleton_utils.del_hook()
    yield
    os.environ.pop("USE_SMDEBUG", None)
    os.environ.pop("SM_HPS", None)
    reset_config_validator()


@patch("smdebug.core.config_validator.is_framework_version_supported", return_value=False)
def test_supported_pytorch_version():
    smdebug.pytorch.singleton_utils.del_hook()
    hook = smdebug.pytorch.singleton_utils.get_hook()
    assert hook is None


@pytest.mark.parametrize(smp_config, ['{"mp_parameters":{"partitions": 2}}', "{}"])
def test_disabling_detailed_profiler(simple_profiler_config_parser, smp_config=None):
    os.environ["SM_HPS"] = smp_config
    smdebug.pytorch.singleton_utils.get_hook()
    assert (
        get_config_validator("pytorch").autograd_profiler_supported
        == "mp_parameters"
        not in smp_config
    )
