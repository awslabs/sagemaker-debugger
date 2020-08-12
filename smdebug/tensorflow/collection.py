# Third Party
import tensorflow.compat.v1 as tf
from tensorflow.python.distribute import values

# First Party
from smdebug.core.collection import DEFAULT_TF_COLLECTIONS
from smdebug.core.collection import Collection as BaseCollection
from smdebug.core.collection import CollectionKeys
from smdebug.core.collection_manager import CollectionManager as BaseCollectionManager
from smdebug.core.logger import get_logger

# Local
from .tensor_ref import TensorRef
from .utils import is_tf_version_2x

logger = get_logger()


class Collection(BaseCollection):
    def __init__(
        self,
        name,
        include_regex=None,
        tensor_names=None,
        reduction_config=None,
        save_config=None,
        save_histogram=True,
    ):
        super().__init__(
            name, include_regex, tensor_names, reduction_config, save_config, save_histogram
        )
        # Different  graphs can have same tensor name, so we store
        # _tensors map for each individual graph
        # value is mapping from tf_name to TensorInCollection object
        self._graph_tensors_map = {}

    def _store_tensor_ref(self, tensor_ref):
        if tensor_ref:
            self.set_tensor_ref(tensor_ref)
        return tensor_ref

    def add_variable(self, arg, export_name=None, mode=None, original_tensor=None):
        # in keras we need to store the mode and only get tensors by mode
        tensor_ref = TensorRef.from_variable(
            arg, export_name=export_name, mode=mode, original_tensor=original_tensor
        )
        return self._store_tensor_ref(tensor_ref)

    def add_distributed_variable(self, arg, export_name=None, mode=None):
        # in keras we need to store the mode and only get tensors by mode
        tensors = []
        for value in arg._values:
            if export_name is None:
                export_name = arg.name
            tensors.append(
                self.add_variable(value, export_name=export_name, mode=mode, original_tensor=arg)
            )
        return tensors

    def add_aggregating_variable(self, arg, name=None, mode=None):
        return self.add_variable(arg.get(), name, mode=mode)

    def add_tensor(self, arg, name=None, mode=None):
        # in keras we need to store the mode and only get tensors by mode
        return self._store_tensor_ref(TensorRef.from_tensor(arg, name, mode=mode))

    def add_operation(self, arg, mode=None):
        return [self.add_tensor(t, mode=mode) for t in arg.outputs]

    def add(self, arg):
        self.add_for_mode(arg)

    def add_for_mode(self, arg, mode=None):
        """
         Adds tensors to the collection from a given Operation, Tensor, Variable or MirroredVariable
         :param arg: the argument to add to collection
         """
        get_logger().debug(f"type:{type(arg)} arg:{arg} self:{id(self)}")
        if isinstance(arg, list) or isinstance(arg, set):
            for a in arg:
                get_logger().debug(f"Adding for list type:{type(a)} self:{id(self)}")
                self.add_for_mode(a, mode)
        elif isinstance(arg, tf.Operation):
            return self.add_operation(arg, mode=mode)
        elif isinstance(arg, tf.Variable):
            return self.add_variable(arg, mode=mode)
        elif isinstance(arg, tf.Tensor):
            return self.add_tensor(arg, mode=mode)
        elif isinstance(arg, values.DistributedValues):
            return self.add_distributed_variable(arg, mode=mode)
        elif isinstance(arg, values.AggregatingVariable):
            return self.add_aggregating_variable(arg, mode=mode)
        else:
            logger.warning(
                f"Could not add {arg} of type {arg.__class__} to collection {self.name}."
                "Add can only take tf.Operation, tf.Variable, tf.Tensor, "
                "tf.MirroredVariable and list or set of any of the above."
            )

    def _get_tensor_map_from_graph_tensor_map(self, graph):
        if graph is None:
            graph = tf.get_default_graph()
        if graph in self._graph_tensors_map:
            return self._graph_tensors_map[graph]
        else:
            return {}

    def get_tensors_dict(self, graph=None):
        return self._get_tensor_map_from_graph_tensor_map(graph)

    def get_tensors(self, mode=None, graph=None):
        tensor_map = self._get_tensor_map_from_graph_tensor_map(graph)
        tensors = tensor_map.values()
        if mode is not None:
            tensors = [t for t in tensors if mode in t.modes]
        if graph is not None:
            tensors = [t for t in tensors if graph == t.tf_obj.graph]
        return tensors

    def get_export_names_of_tensors(self):
        return self.tensor_names

    def get_tensor(self, name, graph=None):
        tensor_map = self._get_tensor_map_from_graph_tensor_map(graph)
        if name in tensor_map:
            return tensor_map[name]
        else:
            return None

    def set_tensor_ref(self, tensor, tensor_name: tf.Tensor = None, graph=None):
        """
        Map tf_obj to a name.
        In case of EagerTensor, rely on the tensor name
        passed by the caller.
        :param tensor: tf_obj or EagerTensor
        :param tensor_name: name of EagerTensor
        """
        # should always be a mapping from tf_obj.name to the argument
        if tensor_name:
            name = export_name = tensor_name
        else:
            name = tensor.name
            export_name = tensor.export_name
        get_logger().debug(f"In set_tensor_ref: tensor_name:{name}")
        if graph is None:
            graph = tf.get_default_graph()
        if graph not in self._graph_tensors_map:
            self._graph_tensors_map[graph] = {}
        if name not in self._graph_tensors_map[graph]:
            self._graph_tensors_map[graph][name] = tensor
            get_logger().debug(
                f"Added name:{name} to graph:{graph} to tensor:{tensor} self:{id(self)} collection_name:{self.name}"
            )

        self.add_tensor_name(export_name)
        # including exact name in collection regex, this is to ensure that we always include this tensor name if it
        # is present in graph from now onwards. For estimator, graph changes between train and evaluation phase,
        # this will ensure that tensors will get captured in evaluation phase also, if tensor names are present in
        # evaluation graph
        if name == export_name:
            self.include("^" + name + "$")
        else:
            get_logger().debug(
                f"Export_name:{export_name} != name:{name} . Not adding to include in collection collection_name:{self.name} selfId:{id(self)}"
            )
            self.include("^" + export_name + "$")

    def has_tensor(self, name, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        # tf object name
        return graph in self._graph_tensors_map and name in self._graph_tensors_map[graph]

    def add_keras_layer(self, layer, inputs=False, outputs=True):
        if inputs:
            input_tensor_regex = layer.name + "/inputs/"
            self.include(input_tensor_regex)
        if outputs:
            output_tensor_regex = layer.name + "/outputs/"
            self.include(output_tensor_regex)


class CollectionManager(BaseCollectionManager):
    def __init__(self, collections=None, create_default=True):
        super().__init__(collections=collections)
        if create_default:
            for n in DEFAULT_TF_COLLECTIONS:
                self.create_collection(n)
            if is_tf_version_2x() and tf.executing_eagerly():
                self.get(CollectionKeys.BIASES).include("^(?!gradient).*bias")
            else:
                self.get(CollectionKeys.BIASES).include("bias")

    def create_collection(self, name):
        super().create_collection(name, cls=Collection)

    @classmethod
    def load(cls, filename, coll_class=Collection):
        return super().load(filename, coll_class)

    @classmethod
    def load_from_string(cls, s, coll_class=Collection):
        return super().load_from_string(s, coll_class)
