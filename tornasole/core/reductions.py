import re
import numpy as np
from tornasole.core.reduction_config import ALLOWED_REDUCTIONS, ALLOWED_NORMS

TORNASOLE_REDUCTIONS_PREFIX = "tornasole/reductions/"


def get_numpy_reduction(reduction_name, numpy_data, abs=False):
    if reduction_name not in ALLOWED_REDUCTIONS and reduction_name not in ALLOWED_NORMS:
        raise ValueError("Invalid reduction type %s" % reduction_name)

    if abs:
        numpy_data = np.absolute(numpy_data)
    return get_basic_numpy_reduction(reduction_name, numpy_data)


def get_basic_numpy_reduction(reduction_name, numpy_data):
    if reduction_name in ALLOWED_REDUCTIONS:
        if reduction_name in ["min", "max"]:
            return getattr(np, "a" + reduction_name)(numpy_data)
        elif reduction_name in ["mean", "prod", "std", "sum", "variance"]:
            if reduction_name == "variance":
                reduction_name = "var"
            return getattr(np, reduction_name)(numpy_data)
    elif reduction_name in ALLOWED_NORMS:
        if reduction_name in ["l1", "l2"]:
            ord = int(reduction_name[1])
        else:
            ord = None

        if abs:
            rv = np.linalg.norm(np.absolute(numpy_data), ord=ord)
        else:
            rv = np.linalg.norm(numpy_data, ord=ord)
        return rv
    return None


def get_reduction_tensor_name(tensorname, reduction_name, abs):
    tname = re.sub(r":\d+", "", f"{reduction_name}/{tensorname}")
    if abs:
        tname = "abs_" + tname
    tname = TORNASOLE_REDUCTIONS_PREFIX + tname
    return tname


def reverse_reduction_tensor_name(reduction_tensor_name):
    rest = reduction_tensor_name.split(TORNASOLE_REDUCTIONS_PREFIX)[1]
    parts = rest.split("/", 1)
    reduction_name = parts[0]
    if "abs_" in reduction_name:
        abs = True
        reduction_op_name = reduction_name.split("abs_")[1]
    else:
        abs = False
        reduction_op_name = reduction_name
    tensor_name = parts[1]
    return tensor_name, reduction_op_name, abs
