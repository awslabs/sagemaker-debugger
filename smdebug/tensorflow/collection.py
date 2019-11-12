# Third Party
import tensorflow as tf
from tensorflow.python.distribute import values

# First Party
from smdebug.core.collection import Collection as BaseCollection
from smdebug.core.collection import CollectionKeys
from smdebug.core.collection_manager import CollectionManager as BaseCollectionManager
from smdebug.core.logger import get_logger

# Local
from .tensor_ref import TensorRef, TensorType

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
        # mapping from tf_name to TensorInCollection object
        self._tensors = {}

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

    def add_mirrored_variable(self, arg, export_name=None, mode=None):
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
        if isinstance(arg, list) or isinstance(arg, set):
            for a in arg:
                self.add_for_mode(a, mode)
        elif isinstance(arg, tf.Operation):
            return self.add_operation(arg, mode=mode)
        elif isinstance(arg, tf.Variable):
            return self.add_variable(arg, mode=mode)
        elif isinstance(arg, tf.Tensor):
            return self.add_tensor(arg, mode=mode)
        elif isinstance(arg, values.MirroredVariable):
            return self.add_mirrored_variable(arg, mode=mode)
        elif isinstance(arg, values.AggregatingVariable):
            return self.add_aggregating_variable(arg, mode=mode)
        else:
            logger.error(
                f"Could not add {arg} of type {arg.__class__} to collection {self.name}."
                "Add can only take tf.Operation, tf.Variable, tf.Tensor, "
                "tf.MirroredVariable and list or set of any of the above."
            )

    def add_reduction_tensor(self, tensor, original_tensor, export_name=None):
        tensor_ref = TensorRef.create_reduction(tensor, original_tensor, export_name)
        if tensor_ref:
            self._tensors[tensor_ref.name] = tensor_ref

    def get_tensors_dict(self):
        return self._tensors

    def get_tensors(self, mode=None, graph=None):
        tensors = self._tensors.values()
        if mode is not None:
            tensors = [t for t in tensors if mode in t.modes]
        if graph is not None:
            tensors = [t for t in tensors if graph == t.tf_obj.graph]
        return tensors

    def get_tensor(self, name):
        return self._tensors[name]

    def set_tensor_ref(self, tensor):
        # should always be a mapping from tf_obj.name to the argument
        self._tensors[tensor.name] = tensor
        self.add_tensor_name(tensor.export_name)

    def has_tensor(self, name):
        # tf object name
        return name in self._tensors

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
            for n in [
                CollectionKeys.DEFAULT,
                CollectionKeys.WEIGHTS,
                CollectionKeys.GRADIENTS,
                CollectionKeys.LOSSES,
                CollectionKeys.SCALARS,
                CollectionKeys.METRICS,
                CollectionKeys.INPUTS,
                CollectionKeys.OUTPUTS,
                CollectionKeys.ALL,
            ]:
                self.create_collection(n)
        self._tensors = {}

    def create_collection(self, name):
        super().create_collection(name, cls=Collection)

    @classmethod
    def load(cls, filename, coll_class=Collection):
        return super().load(filename, coll_class)

    @classmethod
    def load_from_string(cls, s, coll_class=Collection):
        return super().load_from_string(s, coll_class)


_collection_manager = CollectionManager()


def reset_collections():
    global _collection_manager
    del _collection_manager
    _collection_manager = CollectionManager()


def add_to_collection(collection_name, args):
    get_collection(collection_name).add(args)


def add_to_default_collection(args):
    add_to_collection(CollectionKeys.DEFAULT, args)


def get_collection(collection_name):
    return _collection_manager.get(collection_name, create=True)


def get_collections():
    return _collection_manager.collections


def export_collections(path):
    if _collection_manager:
        _collection_manager.export(path)


def get_collection_manager():
    return _collection_manager


def load_collections(path):
    global _collection_manager
    _collection_manager = CollectionManager.load(path)
