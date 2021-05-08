# Future
from __future__ import print_function


def test_supported_tensorflow_version():
    import smdebug.pytorch.singleton_utils
    from unittest.mock import patch

    with patch(
        "smdebug.core.config_validator.is_framework_version_supported"
    ) as override_is_current_version_supported:
        smdebug.tensorflow.singleton_utils.del_hook()

        override_is_current_version_supported.return_value = False
        hook = smdebug.tensorflow.singleton_utils.get_hook()
        assert hook == None
