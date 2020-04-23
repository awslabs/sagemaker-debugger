# Standard Library
from re import search

# Third Party
import tensorflow.compat.v2 as tf


def is_tf_2_2():
    """
    TF 2.0 returns ['accuracy', 'batch', 'size'] as metric collections.
    where 'batch' is the batch number and size is the batch size.
    But TF 2.2 returns ['accuracy', 'batch'] in eager mode, reducing the total
    number of tensor_names emitted by 1.
    :return: bool
    """
    if search("2.2..", tf.__version__):
        return True
    return False
