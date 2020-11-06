# Third Party
import numpy as np
import pytest
from tensorflow.python.framework.dtypes import _NP_TO_TF
from tests.tensorflow2.utils import is_tf_2_2

# First Party
from smdebug.core.tfevent.util import _get_proto_dtype


@pytest.mark.skipif(
    is_tf_2_2() is False, reason="Brain Float Is Unavailable in lower versions of TF"
)
def test_tensorflow2_datatypes():
    # _NP_TO_TF contains all the mappings
    # of numpy to tf types
    try:
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
            assert False, f"{_type} not supported"
    assert True
