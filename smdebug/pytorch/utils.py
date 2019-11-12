# Third Party
import numpy as np
import torch

# First Party
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.core.reductions import get_numpy_reduction


def get_reduction_of_data(reduction_name, tensor_data, tensor_name, abs=False):
    if isinstance(tensor_data, np.ndarray):
        return get_numpy_reduction(reduction_name, tensor_data, abs)
    if abs:
        tensor_data = torch.abs(tensor_data)

    if reduction_name in ALLOWED_REDUCTIONS:
        if reduction_name == "variance":
            reduction_name = "var"
        assert hasattr(torch.Tensor, reduction_name)
        f = getattr(torch.Tensor, reduction_name)
        op = f(tensor_data)
        return op
    elif reduction_name in ALLOWED_NORMS:
        if reduction_name in ["l1", "l2"]:
            ord = int(reduction_name[1])
        else:
            raise RuntimeError(
                "Invalid normalization operation {0} for torch.Tensor".format(reduction_name)
            )
        op = torch.norm(tensor_data, p=ord)
        return op
    elif hasattr(torch, reduction_name):
        f = getattr(torch, reduction_name)
        op = f(tensor_data)
        return op
    raise RuntimeError("Invalid reduction_name {0}".format(reduction_name))


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
        raise TypeError(
            "_make_numpy_array only accepts input types of numpy.ndarray, scalar,"
            " and Torch Tensor, while received type {}".format(str(type(x)))
        )
