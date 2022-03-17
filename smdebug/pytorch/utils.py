# Third Party
# Standard Library
from functools import lru_cache

import numpy as np
import torch
from packaging import version

# First Party
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.core.reductions import get_numpy_reduction

# Cached Pytorch Version
PT_VERSION = version.parse(torch.__version__)

SUPPORTED_PT_VERSION_THRESHOLD = version.parse("1.12.0")


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


@lru_cache(maxsize=1)
def is_pt_1_5():
    """
    Determine whether the version of torch is 1.5.x
    :return: bool
    """
    return version.parse("1.5.0") <= PT_VERSION < version.parse("1.6.0")


@lru_cache(maxsize=1)
def is_pt_1_6():
    """
    Determine whether the version of torch is 1.6.x
    :return: bool
    """
    return version.parse("1.6.0") <= PT_VERSION < version.parse("1.7.0")


@lru_cache(maxsize=1)
def is_pt_1_7():
    """
    Determine whether the version of torch is 1.7.x
    :return: bool
    """
    return version.parse("1.7.0") <= PT_VERSION < version.parse("1.8.0")


@lru_cache(maxsize=1)
def is_pt_1_8():
    """
    Determine whether the version of torch is 1.8.x
    :return: bool
    """
    return version.parse("1.8.0") <= PT_VERSION < version.parse("1.9.0")


@lru_cache(maxsize=1)
def is_pt_1_9():
    """
    Determine whether the version of torch is 1.9.x
    :return: bool
    """
    return version.parse("1.9.0") <= PT_VERSION < version.parse("1.10.0")


def is_current_version_supported(pytorch_version=torch.__version__):
    return version.parse("1.5.0") <= version.parse(pytorch_version) < SUPPORTED_PT_VERSION_THRESHOLD
