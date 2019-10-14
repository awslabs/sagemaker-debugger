from tornasole.core.collection import CollectionKeys
from tornasole.core.collection_manager import (
    CollectionManager as BaseCollectionManager,
)  # noqa: E501


class CollectionManager(BaseCollectionManager):
    def __init__(self, create_default=True):
        super().__init__(create_default=create_default)
        if create_default:
            self._register_default_collections()

    def _register_default_collections(self):
        self.get(CollectionKeys.METRIC).include("^[a-zA-z]+-[a-zA-z0-9]+$")
        self.get(CollectionKeys.PREDICTIONS).include("^predictions$")
        self.get(CollectionKeys.LABELS).include("^labels$")
        self.get(CollectionKeys.FEATURE_IMPORTANCE).include("^.*/feature_importance$")
        self.get(CollectionKeys.AVERAGE_SHAP).include("^((?!bias).)*/average_shap$")


_collection_manager = CollectionManager()


def load_collections(path):

    global _collection_manager
    _collection_manager = CollectionManager.load(path)


def reset_collections():

    global _collection_manager
    del _collection_manager
    _collection_manager = CollectionManager()  # noqa: F841


def add_to_collection(collection_name, args):

    get_collection(collection_name).add(args)


def get_collection_manager():

    return _collection_manager


def add_to_default_collection(args):

    add_to_collection(CollectionKeys.DEFAULT, args)


def get_collection(collection_name):

    return _collection_manager.get(collection_name, create=True)


def get_collections():

    return _collection_manager.collections
