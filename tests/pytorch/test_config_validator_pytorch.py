# Future
from __future__ import print_function

# Third Party
import pytest
from packaging import version


@pytest.fixture()
def pytorch_framework_override(monkeypatch):
    import smdebug.pytorch.utils

    monkeypatch.setattr(smdebug.pytorch.utils, "PT_VERSION", version.parse("1.14"))
    return


"""
def test_supported_pytorch_version(pytorch_framework_override, out_dir):
    import smdebug.pytorch.singleton_utils

    hook = smdebug.pytorch.singleton_utils.get_hook()
    assert hook == None
"""


def test_supported_pytorch_version(monkeypatch):
    import smdebug.pytorch.singleton_utils

    monkeypatch.setattr(smdebug.pytorch.utils, "PT_VERSION", version.parse("1.14"), raising=True)
    pt_version = smdebug.pytorch.utils.PT_VERSION
    print(f"Setting the PT_VERSION to {pt_version}")
    hook = smdebug.pytorch.singleton_utils.get_hook()
    assert hook == None
