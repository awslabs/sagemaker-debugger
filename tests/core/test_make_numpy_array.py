# Third Party
import numpy as np

# First Party
from smdebug.core.utils import make_numpy_array


def test_make_numpy_array():
    simple_numpy_array = np.ndarray(shape=(2, 2), dtype=float, order="F")

    # Check support for ndarray
    try:
        x = make_numpy_array(simple_numpy_array)
        assert x.all() == simple_numpy_array.all()
    except:
        assert False

    # Check support for scalar
    simple_scalar = "foo"
    try:
        x = make_numpy_array(simple_scalar)
        assert x == np.array([simple_scalar])
    except:
        assert False

    # Check support for tuple
    simple_tuple = (0.5, 0.7)
    try:
        x = make_numpy_array(simple_tuple)
        assert x.all() == np.array(simple_tuple).all()
    except:
        assert False

    # Check support for list
    simple_list = [0.5, 0.7]
    try:
        x = make_numpy_array(simple_list)
        assert x.all() == np.array(simple_list).all()
    except:
        assert False

    # Check support for dict
    simple_dict = {"a": 0.5, "b": 0.7}
    try:
        x = make_numpy_array(simple_dict)
        assert x == np.array(simple_dict)
    except:
        assert False
