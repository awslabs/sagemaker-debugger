# Third Party
from tensorflow.python.framework.dtypes import _NP_TO_TF

# First Party
from smdebug.core.tfevent.util import _get_proto_dtype


def test_tensorflow2_datatypes():
    # _NP_TO_TF contains all the mappings
    # of numpy to tf types
    for _type in _NP_TO_TF:
        try:
            _get_proto_dtype(_type)
        except KeyError:
            assert False
    assert True
