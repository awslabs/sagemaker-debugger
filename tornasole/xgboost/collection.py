from tornasole.core.collection import Collection
from tornasole.core.collection_manager import CollectionManager as BaseCollectionManager  # noqa: E501


class CollectionManager(BaseCollectionManager):
    def __init__(self, create_default=True):
        super().__init__()
        if create_default:
            self._register_default_collections()

    def create_collection(self, name):
        self.collections[name] = Collection(name)

    def _register_default_collections(self):
        metric_collection = Collection(
            "metric",
            include_regex=["^[a-zA-z]+-[a-zA-z0-9]+$"])
        predictions_collection = Collection(
            "predictions",
            include_regex=["^predictions$"])
        labels_collection = Collection(
            "labels",
            include_regex=["^labels$"])
        feat_imp_collection = Collection(
            "feature_importance",
            include_regex=["^.*/feature_importance$"])
        shap_collection = Collection(
            "average_shap",
            include_regex=["^((?!bias).)*/average_shap$"])
        self.add(metric_collection)
        self.add(predictions_collection)
        self.add(labels_collection)
        self.add(feat_imp_collection)
        self.add(shap_collection)


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

    add_to_collection('default', args)


def get_collection(collection_name):

    try:
        c = _collection_manager.get(collection_name)
    except KeyError:
        _collection_manager.create_collection(collection_name)
        c = _collection_manager.get(collection_name)
    return c


def get_collections():

    return _collection_manager.collections
