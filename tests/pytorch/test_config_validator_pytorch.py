# Future
from __future__ import print_function


def test_supported_pytorch_version():
    import smdebug.pytorch.singleton_utils
    from unittest.mock import patch

    with patch(
        "smdebug.core.config_validator.is_framework_version_supported"
    ) as override_is_current_version_supported:
        smdebug.pytorch.singleton_utils.del_hook()

        override_is_current_version_supported.return_value = False
        hook = smdebug.pytorch.singleton_utils.get_hook()
        assert hook == None
    # Disengaging the hook also sets the environment variable USE_SMDEBUG to False, we would need to reset this
    # variable for further tests.
    import os

    del os.environ["USE_SMDEBUG"]


def test_disabling_detailed_profiler(simple_profiler_config_parser):
    import smdebug.pytorch.singleton_utils
    from smdebug.core.config_validator import get_config_validator, reset_config_validator

    smdebug.pytorch.singleton_utils.del_hook()
    reset_config_validator()
    import os

    os.environ["SM_HPS"] = '{"mp_parameters":{"partitions": 2}}'
    smdebug.pytorch.singleton_utils.get_hook()
    assert get_config_validator("pytorch").autograd_profiler_supported == False

    os.environ["SM_HPS"] = "{}"
    smdebug.pytorch.singleton_utils.del_hook()
    reset_config_validator()
    assert get_config_validator("pytorch").autograd_profiler_supported == True
    del os.environ["SM_HPS"]
