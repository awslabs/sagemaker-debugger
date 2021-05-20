# Standard Library
import collections
import json
from enum import Enum
from typing import Callable, List, Optional

# Third Party
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from packaging import version
from tensorflow.python.distribute import values

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.utils import error_handling_agent

# Cached TF Version
TF_VERSION = version.parse(tf.__version__)


def does_tf_support_mixed_precision_training():
    # The Keras mixed precision API is first available in TensorFlow 2.1.0
    # See: https://www.tensorflow.org/guide/mixed_precision
    return TF_VERSION >= version.parse("2.1.0")


def supported_tf_variables():
    if does_tf_support_mixed_precision_training():
        if is_tf_version_greater_than_2_4_x():
            # tensorflow mixed preicison api is experimental in versions below 2.4.0
            from tensorflow.python.keras.mixed_precision import autocast_variable

        else:
            from tensorflow.python.keras.mixed_precision.experimental import autocast_variable

        return tf_v1.Variable, autocast_variable.AutoCastVariable
    else:
        return tf_v1.Variable


class ModelOutput:
    LABELS = "smdebug_y"
    PREDICTIONS = "smdebug_y_pred"
    VAL_LABELS = "val_smdebug_y"
    VAL_PREDICTIONS = "val_smdebug_y_pred"


ModelOutputs = {
    ModelOutput.LABELS,
    ModelOutput.PREDICTIONS,
    ModelOutput.VAL_LABELS,
    ModelOutput.VAL_PREDICTIONS,
}


def get_model_output_export_name(key):
    export_names = {
        ModelOutput.PREDICTIONS: "predictions",
        ModelOutput.LABELS: "labels",
        ModelOutput.VAL_LABELS: "labels",
        ModelOutput.VAL_PREDICTIONS: "predictions",
    }
    return export_names[key]


class ModelInput:
    INPUTS = "smdebug_x"


ModelInputs = {ModelInput.INPUTS}


def get_model_input_export_name():
    return f"model_input"


class TFDistributionStrategy(Enum):
    NONE = 0
    HOROVOD = 1
    MIRRORED = 2
    PARAMETER_SERVER = 3
    SMDATAPARALLEL = 4
    UNSUPPORTED = 100


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


def tensor_can_be_saved(root_tensor, subgraph_nodes, unfilled_placeholders):
    """
    If a tensor x depends on an unfilled placeholder, then it can't be saved and should be skipped.
    This 4th step is done by performing BFS from this tensor x, and going up
    its inputs for any node which is not in the subgraph.

    If a node reached through this BFS is not in the subgraph
    and is an unfilled placeholder, then the tensor x can't be saved.

    :param root_tensor: the tensor from which to start BFS
    :param subgraph_nodes: the subgraph which can reach the current fetches
    :param unfilled_placeholders: placeholders which were not assigned values
    :return:
    """
    seen, queue = {root_tensor}, collections.deque([root_tensor])
    while queue:
        tensor = queue.popleft()
        if tensor.op.name not in subgraph_nodes:
            if len(tensor.op.inputs) == 0 and tensor in unfilled_placeholders:
                # current tensor is not in the subgraph,
                # but it also has no inputs which might be in the subgraph
                # this means tf_tensor is not connected the fetches through the subgraph
                return False
            for ti in tensor.op.inputs:
                if ti not in seen:
                    seen.add(ti)
                    queue.append(ti)
    return True


def build_fetches_tuple(fetches):
    if (
        not isinstance(fetches, list)
        and not isinstance(fetches, tuple)
        and not isinstance(fetches, dict)
    ):
        fetches = [fetches]
    original_fetch_ops = get_original_fetch_ops(fetches)
    # sorting to create a unique tuple for lists of all orders
    original_fetch_ops.sort(key=lambda x: x.name)
    # creating a tuple as we need a immutable var for it to server
    # as key into a dictionary
    original_fetch_ops_tuple = tuple(original_fetch_ops)
    return original_fetch_ops_tuple


def get_original_fetch_ops(fetches):
    if isinstance(fetches, tf.Tensor) or isinstance(fetches, tf.Variable):
        return [fetches.op]
    elif isinstance(fetches, tf.Operation):
        return [fetches]
    elif isinstance(fetches, values.Mirrored):
        return [x.op for x in fetches.values]
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
    elif fetches is None:
        return []
    else:
        raise RuntimeError("Invalid fetches")


""""
The TF_CONFIG environment variable is the standard way to specify the cluster configuration
to each worker that is part of the cluster.


Given below some examples of TF_CONFIG:


  Example of `TF_CONFIG` for chief training worker (must have one and only one):

  Note that the chief worker also does the model training job, similar to other
  non-chief training workers (see next paragraph). In addition to the model
  training, it manages some extra work, e.g., checkpoint saving and restoring,
  writing summaries, etc.

  TF_CONFIG='{
      "cluster": {
          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "chief", "index": 0}
  }'


  Example of `TF_CONFIG` for non-chief training worker (optional, could be
  multiple):

  TF_CONFIG='{
      "cluster": {
          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "worker", "index": 0}
  }'

  where the `task.index` should be set as 0, 1, 2, in this example, respectively
  for non-chief training workers.


  Example of `TF_CONFIG` for parameter server, aka ps (could be multiple):

  TF_CONFIG='{
      "cluster": {
          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "ps", "index": 0}
  }'

  where the `task.index` should be set as 0 and 1, in this example, respectively
  for parameter servers.

  Example of `TF_CONFIG` for evaluator task. Evaluator is a special task that is
  not part of the training cluster. There could be only one. It is used for
  model evaluation.

  TF_CONFIG='{
      "cluster": {

          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "evaluator", "index": 0}
  }'

  NOTE: If the "chief" is missing in TF_CONFIG["cluster"], the worker with index 0 assumes this role.

See https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config_environment_variable
"""


def load_tf_config_json(tf_config: str):
    try:
        return json.loads(tf_config)
    except (json.JSONDecodeError, TypeError):
        # if tf_config is None throws TypeError, so return None from next line
        return None


def is_parameter_server_strategy(tf_config_json: dict) -> bool:
    try:
        return "cluster" in tf_config_json and "ps" in tf_config_json["cluster"]
    except TypeError:
        # when json is None
        return False


def get_worker_id_from_tf_config(tf_config_json: dict) -> str:
    """Valid roles in a cluster is "chief", "worker", "ps" and "evaluator"."""
    task = tf_config_json["task"]
    worker_type = task["type"]
    worker_index = task["index"]
    return f"{worker_type}_{worker_index}"


def get_num_workers_from_tf_config(tf_config_json: dict) -> int:
    workers = tf_config_json["cluster"]["worker"]
    if "chief" in tf_config_json["cluster"]:
        workers.extend(tf_config_json["cluster"]["chief"])
    return len(workers)


def get_chief_worker_from_tf_config(tf_config_json: dict):
    if "chief" in tf_config_json["cluster"]:
        return "chief_0"
    else:
        raise NotImplementedError
        # todo


def is_mirrored_strategy(strat):
    return isinstance(
        strat, (tf.distribute.MirroredStrategy, tf.compat.v1.distribute.MirroredStrategy)
    )


def is_keras_optimizer(obj):
    for cls in obj.__class__.__mro__:
        if ".".join([cls.__module__, cls.__name__]) == "keras.optimizers.Optimizer":
            return True
    return False


def get_export_name_for_keras(layer, tensor_type, tensor=None):
    if tensor_type in ["input", "output", "weight"]:
        if isinstance(layer, str):
            # Tensor.name is meaningless when eager execution is enabled.
            return f"{layer}/{tensor_type}s"
        else:
            return f"{layer.name}/{tensor_type}s/{tensor.name}"
    else:
        return None


def get_keras_layer_inputs(layer):
    # will throw an exception if _inbound_nodes is not loaded
    layer.get_input_at(0)
    input_tensors = []
    for idx in range(len(layer._inbound_nodes)):
        inputs = layer.get_input_at(idx)
        if not isinstance(inputs, list):
            inputs = [inputs]
        for input_index, inp in enumerate(inputs):
            input_tensors.append(inp)
    return input_tensors


class LayerWithHooks(tf.keras.layers.Layer):
    def __init__(
        self,
        layer: tf.keras.layers.Layer,
        hooks: List[Callable[[tf.Tensor, tf.Tensor], Optional[tf.Tensor]]] = None,
    ):
        super().__init__()
        self._layer = layer
        self._hooks = hooks or []

    def call(self, input: tf.Tensor) -> tf.Tensor:
        output = self._layer(input)
        for hook in self._hooks:
            hook_result = hook(input, output)
            if hook_result is not None:
                output = hook_result
        return output

    def register_hook(self, hook: Callable[[tf.Tensor, tf.Tensor], Optional[tf.Tensor]]) -> None:
        self._hooks.append(hook)


class InputOutputSaver:
    def __init__(self):
        self.layer_input = None
        self.layer_output = None

    def __call__(self, inputs, *args, **kwargs) -> None:
        self.layer_input = kwargs["layer_input"]
        self.layer_output = kwargs["layer_output"]


def get_layer_call_fn(layer: tf.keras.layers.Layer) -> Callable[[tf.Tensor], tf.Tensor]:
    old_call_fn = layer.call
    layer.old_call = old_call_fn

    @error_handling_agent.catch_smdebug_errors(default_return_val=old_call_fn)
    def call(inputs, *args, **kwargs) -> tf.Tensor:
        layer_input = inputs
        layer_output = old_call_fn(inputs, *args, **kwargs)
        for hook in layer._hooks:
            hook_result = hook(inputs, layer_input=layer_input, layer_output=layer_output)
            if hook_result is not None:
                layer_output = hook_result
        return layer_output

    return call


def get_non_device_tensors(tensor_refs):
    non_dev_tensors = []
    for tensor_ref in tensor_refs:
        if not tensor_ref.tf_obj.device:
            non_dev_tensors.append(tensor_ref)
    return non_dev_tensors


def get_keras_layer_outputs(layer):
    # will throw an exception if _inbound_nodes is not loaded
    layer.get_output_at(0)
    output_tensors = []
    for idx in range(len(layer._inbound_nodes)):
        outputs = layer.get_output_at(idx)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for output_index, outp in enumerate(outputs):
            output_tensors.append(outp)
    return output_tensors


def get_keras_mode(mode):
    # Should never be called in TF 1.13 where this is not available
    from tensorflow.python.keras.utils.mode_keys import ModeKeys as KerasModeKeys

    if mode == ModeKeys.TRAIN:
        return KerasModeKeys.TRAIN
    elif mode == ModeKeys.EVAL:
        return KerasModeKeys.TEST
    elif mode == ModeKeys.PREDICT:
        return KerasModeKeys.PREDICT


def is_tf_version_2x():
    return TF_VERSION >= version.parse("2.0.0")


def is_tf_version_2_2_x():
    return version.parse("2.2.0") <= TF_VERSION < version.parse("2.3.0")


def is_tf_version_2_3_x():
    return version.parse("2.3.0") <= TF_VERSION < version.parse("2.4.0")


def is_tf_version_2_4_x():
    return version.parse("2.4.0") <= TF_VERSION < version.parse("2.5.0")


def is_tf_version_1_15_x():
    return version.parse("1.15.0") <= TF_VERSION < version.parse("2.0.0")


def is_tf_version_greater_than_2_4_x():
    return version.parse("2.4.0") <= TF_VERSION


def is_profiler_supported_for_tf_version():
    # Profiler Support Added For TF Versions 2.2.0 And Greater
    return version.parse("2.2.0") <= TF_VERSION


def is_current_version_supported(tf_version=tf.__version__):
    return version.parse("1.15.0") <= version.parse(tf_version) < version.parse("2.5.0")
