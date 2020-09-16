# Third Party
import numpy as np
from packaging import version
from tensorflow.python.framework.dtypes import _NP_TO_TF

# First Party
from smdebug.core.tfevent.util import _get_proto_dtype


def test_tensorflow2_datatypes():
    # _NP_TO_TF contains all the mappings
    # of numpy to tf types
    try:
        from tensorflow import __version__ as tf_version

        if version.parse(tf_version) >= version.parse("2.0.0"):
            from tensorflow.python import _pywrap_bfloat16

            # TF 2.x.x Implements a Custom Numpy Datatype for Brain Floating Type
            # Which is currently only supported on TPUs
            _np_bfloat16 = _pywrap_bfloat16.TF_bfloat16_type()
            _NP_TO_TF.pop(_np_bfloat16)
    except (ModuleNotFoundError, ValueError, ImportError):
        pass

    for _type in _NP_TO_TF:
        try:
            _get_proto_dtype(np.dtype(_type))
        except Exception:
            assert False
    assert True
