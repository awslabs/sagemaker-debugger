import torch
from tornasole.core.reduction_config import ALLOWED_REDUCTIONS, ALLOWED_NORMS
import numpy as np
from tornasole.core.reductions import get_numpy_reduction


def get_aggregated_data(aggregation_name, tensor_data, tensor_name, abs=False):
    reduction_name = aggregation_name
    if isinstance(tensor_data, np.ndarray):
        return get_numpy_reduction(reduction_name, tensor_data, abs)
    if abs:
        tensor_data = torch.abs(tensor_data)

    if reduction_name in ALLOWED_REDUCTIONS:
        assert hasattr(torch.Tensor, aggregation_name)
        f = getattr(torch.Tensor, aggregation_name)
        op = f(tensor_data)
        return op
    elif reduction_name in ALLOWED_NORMS:
        if aggregation_name in ['l1', 'l2']:
            ord = int(aggregation_name[1])
        else:
            raise RuntimeError("Invalid normalization operation {0} for torch.Tensor".format(reduction_name))
        op = torch.norm(tensor_data, p=ord)
        return op
    elif hasattr(torch, aggregation_name):
        f = getattr(torch, aggregation_name)
        op = f(tensor_data)
        return op
    raise RuntimeError("Invalid aggregation_name {0}".format(aggregation_name))


def make_numpy_array(x):
    if isinstance(x, np.ndarray):
        return x
    elif np.isscalar(x):
        return np.array([x])
    elif isinstance(x, torch.Tensor):
        return x.to(torch.device("cpu")).data.numpy()
    elif isinstance(x, tuple):
        return np.asarray(x, dtype=x.dtype)
    else:
        raise TypeError('_make_numpy_array only accepts input types of numpy.ndarray, scalar,'
                        ' and Torch Tensor, while received type {}'.format(str(type(x))))
