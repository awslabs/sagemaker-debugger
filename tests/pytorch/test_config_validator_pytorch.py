# Future
from __future__ import print_function


def test_supported_pytorch_version():
    import smdebug.pytorch.singleton_utils
    import pytest

    with pytest.MonkeyPatch.context() as m:
        smdebug.pytorch.singleton_utils.del_hook()

        def override_is_current_version_supported():
            return False

        m.setattr(
            smdebug.pytorch.utils,
            "is_current_version_supported",
            override_is_current_version_supported,
            raising=True,
        )
        hook = smdebug.pytorch.singleton_utils.get_hook()
        assert hook == None
