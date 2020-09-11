# Third Party
import mxnet as mx
import numpy as np

# First Party
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.core.reductions import get_numpy_reduction
from smdebug.core.utils import make_numpy_array


def get_reduction_of_data(aggregation_name, tensor_data, tensor_name, abs=False):
    reduction_name = aggregation_name
    # If tensor_data is of np.ndarray type invoke np operators.
    if isinstance(tensor_data, np.ndarray):
        return get_numpy_reduction(reduction_name, tensor_data, abs)
    if abs:
        tensor_data = mx.ndarray.abs(tensor_data)

    if reduction_name in ALLOWED_REDUCTIONS:
        if hasattr(mx.ndarray, aggregation_name):
            f = getattr(mx.ndarray, aggregation_name)
            op = f(tensor_data, name=tensor_name)
        else:
            # If aggregation is not supported by mxnet, we convert data into np.ndarray and invoke numpy operators
            tensor_data_np = make_numpy_array(tensor_data)
            op = get_numpy_reduction(aggregation_name, numpy_data=tensor_data_np, abs=abs)
        return op
    elif reduction_name in ALLOWED_NORMS:
        if reduction_name in ["l1", "l2"]:
            op = mx.ndarray.norm(data=tensor_data, ord=int(reduction_name[1]))
            return op
        else:
            raise RuntimeError(
                "Invalid normalization operation {0} for mx.NDArray".format(reduction_name)
            )
    elif hasattr(mx, reduction_name):
        f = getattr(mx, reduction_name)
        op = f(tensor_data, name=tensor_name)
        return op
    elif hasattr(np, aggregation_name):
        f = getattr(np, aggregation_name)
        op = f(tensor_data)
        return op
    raise RuntimeError("Invalid aggregation_name {0} for mx.NDArray".format(aggregation_name))
