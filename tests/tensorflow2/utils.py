# Standard Library
from enum import Enum

# Third Party
import tensorflow.compat.v2 as tf
from packaging import version

# Cached TF Version
TF_VERSION = version.parse(tf.__version__)


def is_tf_2_2():
    """
    TF 2.0 returns ['accuracy', 'batch', 'size'] as metric collections.
    where 'batch' is the batch number and size is the batch size.
    But TF 2.2 returns ['accuracy', 'batch'] in eager mode, reducing the total
    number of tensor_names emitted by 1.
    :return: bool
    """
    if TF_VERSION >= version.parse("2.2.0"):
        return True
    return False


def is_tf_2_3():
    if TF_VERSION == version.parse("2.3.0"):
        return True
    return False


def is_tf_version_greater_than_2_4_x():
    return version.parse("2.4.0") <= TF_VERSION


class ModelType(Enum):
    SEQUENTIAL = 0
    FUNCTIONAL = 1
    SUBCLASSED = 2
