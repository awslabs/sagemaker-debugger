import tensorflow.compat.v1 as tf
from tornasole.core.reduction_config import ALLOWED_REDUCTIONS, ALLOWED_NORMS


def get_tensorflow_reduction(reduction_name,
                             tensor, tensor_name, abs=False):
    if reduction_name in ['std', 'variance', 'l1', 'l2']:
        # these reductions create a name with a weird suffix like squeeze or sqrt
        # even if we pass the name we want to the op
        # so we are using a random name first, then
        # using identity op to rename it how we want it
        temp_tensor_name = ''
    else:
        temp_tensor_name = tensor_name

    if reduction_name in ALLOWED_REDUCTIONS:
        f = getattr(tf.math, 'reduce_' + reduction_name)
        if abs:
            op = f(tf.abs(tensor), name=temp_tensor_name)
        else:
            op = f(tensor, name=temp_tensor_name)
    elif reduction_name in ALLOWED_NORMS:
        if reduction_name in ['l1', 'l2']:
            ord = int(reduction_name[1])
        else:
            ord = reduction_name
        if abs:
            op = tf.norm(tf.abs(tensor), ord=ord, name=temp_tensor_name)
        else:
            op = tf.norm(tensor, ord=ord, name=temp_tensor_name)
    else:
        raise RuntimeError(f'Invalid reduction name {reduction_name}')

    if temp_tensor_name != tensor_name:
        op = tf.identity(op, name=tensor_name)
    return op
