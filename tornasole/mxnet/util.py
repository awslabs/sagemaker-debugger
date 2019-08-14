import mxnet as mx
import numpy as np
from mxnet.ndarray import NDArray
from tornasole.core.reduction_config import ALLOWED_REDUCTIONS, ALLOWED_NORMS
from tornasole.core.reductions import get_numpy_reduction


def get_aggregated_data(aggregation_name,
                        tensor_data, tensor_name, abs=False):
    reduction_name = aggregation_name
    if isinstance(tensor_data, np.ndarray):
        return get_numpy_reduction(reduction_name,
                               tensor_data, abs)
    if abs:
        tensor_data = mx.ndarray.abs(tensor_data)

    if reduction_name in ALLOWED_REDUCTIONS:
        assert hasattr(mx.ndarray, aggregation_name)
        f = getattr(mx.ndarray, aggregation_name)
        op = f(tensor_data, name=tensor_name)
        return op
    elif reduction_name in ALLOWED_NORMS:
        if reduction_name is "l1":
            op = mx.ndarray.norm(data=tensor_data, ord=1)
            return op
        elif reduction_name is "l2":
            op = mx.ndarray.norm(data=tensor_data, ord=2)
            return op
        else:
            raise RuntimeError("Invalid normalization operation {0} for mx.NDArray".format(reduction_name))
    elif hasattr(mx, reduction_name):
        f = getattr(mx, reduction_name)
        op = f(tensor_data, name=tensor_name)
        return op
    raise RuntimeError("Invalid aggregation_name {0} for mx.NDArray".format(aggregation_name))


def make_numpy_array(x):
    if isinstance(x, np.ndarray):
        return x
    elif np.isscalar(x):
        return np.array([x])
    elif isinstance(x, NDArray):
        return x.asnumpy()
    elif isinstance(x, tuple):
        # todo: fix this, will crash
        return np.asarray(x, dtype=x.dtype)
    else:
        raise TypeError('_make_numpy_array only accepts input types of numpy.ndarray, scalar,'
                        ' and MXNet NDArray, while received type {}'.format(str(type(x))))