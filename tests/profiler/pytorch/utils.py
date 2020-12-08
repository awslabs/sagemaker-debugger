# Third Party
import torch
from packaging import version


def is_pt_1_5():
    """
    Determine whether the version of torch is 1.5.x
    :return: bool
    """
    return version.parse("1.5.0") <= version.parse(torch.__version__) < version.parse("1.6.0")


def is_pt_1_6():
    """
    Determine whether the version of torch is 1.6.x
    :return: bool
    """
    return version.parse("1.6.0") <= version.parse(torch.__version__) < version.parse("1.7.0")


def is_pt_1_7():
    """
    Determine whether the version of torch is 1.7.x
    :return: bool
    """
    return version.parse("1.7.0") <= version.parse(torch.__version__) < version.parse("1.8.0")
