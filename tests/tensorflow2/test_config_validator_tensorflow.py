# Future
from __future__ import print_function


def test_supported_tensorflow_version():
    import smdebug.tensorflow.singleton_utils
    import smdebug.tensorflow.utils
    import pytest

    with pytest.MonkeyPatch.context() as m:
        smdebug.tensorflow.singleton_utils.del_hook()

        def override_is_current_version_supported():
            return False

        m.setattr(
            smdebug.tensorflow.utils,
            "is_current_version_supported",
            override_is_current_version_supported,
            raising=True,
        )
        hook = smdebug.tensorflow.singleton_utils.get_hook()
        assert hook == None
