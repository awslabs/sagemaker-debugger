import tensorflow as tf
from tornasole.core.reduction_config import ALLOWED_REDUCTIONS, ALLOWED_NORMS


def get_tensorflow_reduction(reduction_name, tensor, on_absolute_values=False):
    if reduction_name in ALLOWED_REDUCTIONS:
        f = getattr(tf.math, "reduce_" + reduction_name)
        if on_absolute_values:
            op = f(tf.abs(tensor))
        else:
            op = f(tensor)
    elif reduction_name in ALLOWED_NORMS:
        if reduction_name in ["l1", "l2"]:
            ord = int(reduction_name[1])
        else:
            ord = reduction_name
        if on_absolute_values:
            op = tf.norm(tf.abs(tensor), ord=ord)
        else:
            op = tf.norm(tensor, ord=ord)
    else:
        raise RuntimeError(f"Invalid reduction name {reduction_name}")

    return op
