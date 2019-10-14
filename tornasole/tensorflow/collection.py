from tensorflow.python.distribute import values
import tensorflow.compat.v1 as tf
from tornasole.core.collection import Collection as BaseCollection, CollectionKeys
from tornasole.core.collection_manager import CollectionManager as BaseCollectionManager


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
        self.tensors = []
        # has the new tensors added to graph
        # reduction_tensor_names has the names of original tensors
        # whose reductions these are
        self.reduction_tensors_added = []

    def add(self, arg):
        if isinstance(arg, list) or isinstance(arg, set):
            for a in arg:
                self.add(a)
        elif isinstance(arg, tf.Operation):
            for t in arg.outputs:
                self.add_tensor(t)
        elif isinstance(arg, tf.Variable) or isinstance(arg, tf.Tensor):
            self.add_tensor(arg)
        elif isinstance(arg, values.MirroredVariable):
            for value in arg._values:
                self.add_tensor(value)
        else:
            raise TypeError(
                "Unknown type of argument %s."
                "Add can only take tf.Operation, tf.Variable, tf.Tensor"
                "and list or set of any of the above." % arg
            )

    def add_tensor(self, t):
        self.add_tensor_name(t.name)
        # tf tries to add variables both by tensor and variable.
        # to avoid duplications, we need to check names
        for x in self.tensors:
            if x.name == t.name:
                return
        self.tensors.append(t)

    def add_reduction_tensor(self, t, original_tensor):
        self.add_tensor_name(original_tensor.name)
        # tf tries to add variables both by tensor and variable.
        # to avoid duplications, we need to check names
        for x in self.reduction_tensors_added:
            if x.name == t.name:
                return
        self.reduction_tensors_added.append(t)

    def remove_tensor(self, t):
        # have to compare names because tensors can have variables, \
        # we don't want to end up comparing tensors and variables
        if t.name in self.tensor_names:
            found_index = None
            for i, lt in enumerate(self.tensors):
                if lt.name == t.name:
                    found_index = i

            self.tensor_names.remove(t.name)

            # this can happen when tensors is cleared but tensor names is not cleared
            # because of emptying tensors and reduction_tensors lists in
            # prepare_collections
            if found_index is None:
                raise IndexError("Could not find tensor to remove")
            self.tensors.pop(found_index)


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
