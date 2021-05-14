# Future
from __future__ import print_function

# Standard Library
import os
from unittest.mock import patch

# Third Party
import pytest

# First Party
import smdebug.pytorch.singleton_utils
from smdebug.core.config_validator import ConfigValidator, reset_config_validator
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser


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


@pytest.mark.parametrize("smp_config", ['{"mp_parameters":{"partitions": 2}}', "{}"])
def test_disabling_detailed_profiler(simple_profiler_config_parser, smp_config):
    os.environ["SM_HPS"] = smp_config
    profiler_config_parser = ProfilerConfigParser()
    ConfigValidator.validate_profiler_config(profiler_config_parser)
    assert (
        profiler_config_parser.config.detailed_profiling_config.disabled
        == "mp_parameters"
        in smp_config
    )
