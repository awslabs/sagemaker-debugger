from tensorflow.python.distribute import values
import tensorflow as tf
from .tensor import Tensor, TensorType
from tornasole.core.collection import Collection as BaseCollection, CollectionKeys
from tornasole.core.collection_manager import CollectionManager as BaseCollectionManager
from tornasole.core.logger import get_logger

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
        # mapping from name_in_graph to TensorInCollection object
        self._tensors = {}

    def _store_ts_tensor(self, ts_tensor):
        if ts_tensor:
            self._tensors[ts_tensor.name_in_graph] = ts_tensor
            self.add_tensor_name(ts_tensor.tornasole_name)

    def add_tensor(self, arg, name=None):
        """
         Adds tensors to the collection from a given Operation, Tensor, Variable or MirroredVariable
         :param arg: the argument to add to collection
         :param name: used only if the added object is tensor or variable
         """
        if isinstance(arg, tf.Operation):
            for t in arg.outputs:
                self._store_ts_tensor(Tensor.from_tensor(t))
        elif isinstance(arg, tf.Variable):
            self._store_ts_tensor(Tensor.from_variable(arg, name))
        elif isinstance(arg, tf.Tensor):
            self._store_ts_tensor(Tensor.from_tensor(arg, name))
        elif isinstance(arg, values.MirroredVariable):
            for value in arg._values:
                self._store_ts_tensor(Tensor.from_variable(value))
        elif isinstance(arg, values.AggregatingVariable):
            self._store_ts_tensor(Tensor.from_variable(arg.get()))
        else:
            logger.error(
                f"Could not add {arg} of type {arg.__class__} to collection {self.name}."
                "Add can only take tf.Operation, tf.Variable, tf.Tensor, "
                "tf.MirroredVariable and list or set of any of the above."
            )

    def add(self, arg):
        if isinstance(arg, list) or isinstance(arg, set):
            for a in arg:
                self.add(a)
        elif isinstance(
            arg,
            (
                tf.Tensor,
                tf.Operation,
                tf.Variable,
                values.MirroredVariable,
                values.AggregatingVariable,
            ),
        ):
            self.add_tensor(arg)
        else:
            logger.error(
                f"Could not add {arg} of type {arg.__class__} to collection {self.name}."
                "Add can only take tf.Operation, tf.Variable, tf.Tensor, "
                "tf.MirroredVariable and list or set of any of the above."
            )

    def add_reduction_tensor(self, tensor, original_tensor, tornasole_name=None):
        ts_tensor = Tensor.create_reduction(tensor, original_tensor, tornasole_name)
        if ts_tensor:
            self._tensors[ts_tensor.name_in_graph] = ts_tensor

    def get_tensors_dict(self):
        return self._tensors

    def get_tensors(self):
        return self._tensors.values()

    def get_tensor(self, name):
        return self._tensors[name]

    def set_tensor(self, tensor):
        name = tensor.name_in_graph
        self._tensors[name] = tensor

    def has_tensor(self, name):
        return name in self._tensors


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
            ]:
                self.create_collection(n)

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
