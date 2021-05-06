# Future
from __future__ import print_function

# Third Party
from packaging import version


def test_supported_pytorch_version(monkeypatch):
    import smdebug.pytorch.singleton_utils

    with monkeypatch.context() as m:
        m.setattr(smdebug.pytorch.utils, "PT_VERSION", version.parse("1.14"), raising=True)
        pt_version = smdebug.pytorch.utils.PT_VERSION
        print(f"Setting the PT_VERSION to {pt_version}")
        hook = smdebug.pytorch.singleton_utils.get_hook()
        assert hook == None
