# Future
from __future__ import print_function

# Third Party
from packaging import version


def test_supported_tensorflow_version(monkeypatch):
    import smdebug.tensorflow.singleton_utils
    import smdebug.tensorflow.utils

    with monkeypatch.context() as m:
        m.setattr(smdebug.tensorflow.utils, "TF_VERSION", version.parse("1.1"), raising=True)
        hook = smdebug.tensorflow.singleton_utils.get_hook()
        assert hook == None
