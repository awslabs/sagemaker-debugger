# Future
from __future__ import print_function

# Third Party
import pytest
from packaging import version


@pytest.fixture()
def tensorflow_framework_override(monkeypatch):
    import smdebug.tensorflow.utils

    monkeypatch.setattr(smdebug.tensorflow.utils, "TF_VERSION", version.parse("1.1"))
    return


def test_supported_tensorflow_version(monkeypatch):
    import smdebug.tensorflow.singleton_utils
    import smdebug.tensorflow.utils

    monkeypatch.setattr(smdebug.tensorflow.utils, "TF_VERSION", version.parse("1.1"), raising=True)
    hook = smdebug.tensorflow.singleton_utils.get_hook()
    assert hook == None
