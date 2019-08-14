import tensorflow as tf


def node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]


def extract_graph_summary(graph_def):
  """Extracts useful information from the graph and returns them."""
  name_to_input_name = {}  # Keyed by the dest node name.
  name_to_node = {}  # Keyed by node name.

  # Keeps track of node sequences. It is important to still output the
  # operations in the original order.
  name_to_seq_num = {}  # Keyed by node name.
  seq = 0
  for node in graph_def.node:
    n = node_name(node.name)
    name_to_node[n] = node
    name_to_input_name[n] = [node_name(x) for x in node.input]
    name_to_seq_num[n] = seq
    seq += 1
  return name_to_input_name, name_to_node, name_to_seq_num


def get_original_fetch_ops(fetches):
    if isinstance(fetches, tf.Tensor) or isinstance(fetches, tf.Variable):
        return [fetches.op]
    elif isinstance(fetches, tf.Operation):
        return [fetches]
    elif isinstance(fetches, list):
        rval = []
        for f in fetches:
            rval.extend(get_original_fetch_ops(f))
        return rval
    elif isinstance(fetches, dict):
        rval = []
        for key in fetches:
            rval += get_original_fetch_ops(fetches[key])
        return rval
    else:
        raise RuntimeError('Invalid fetches')


def size_and_shape(t):
    if type(t) == bytes or type(t) == str:
        return (len(t), [len(t)])
    return (t.nbytes, t.shape)
