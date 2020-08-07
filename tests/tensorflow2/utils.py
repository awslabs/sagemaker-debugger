# Standard Library
# Third Party
import tensorflow.compat.v2 as tf
from packaging import version


def is_tf_2_2():
    """
    TF 2.0 returns ['accuracy', 'batch', 'size'] as metric collections.
    where 'batch' is the batch number and size is the batch size.
    But TF 2.2 returns ['accuracy', 'batch'] in eager mode, reducing the total
    number of tensor_names emitted by 1.
    :return: bool
    """
    if version.parse(tf.__version__) == version.parse("2.2.0"):
        return True
    return False


def is_tf_2_3():
    if version.parse(tf.__version__) == version.parse("2.3.0"):
        return True
    return False
