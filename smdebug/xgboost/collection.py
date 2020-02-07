# First Party
from smdebug.core.collection import DEFAULT_XGBOOST_COLLECTIONS, CollectionKeys
from smdebug.core.collection_manager import CollectionManager as BaseCollectionManager


class CollectionManager(BaseCollectionManager):
    def __init__(self, create_default=True):
        super().__init__(create_default=create_default)
        if create_default:
            self._register_default_collections()

    def _register_default_collections(self):
        for c in DEFAULT_XGBOOST_COLLECTIONS:
            self.create_collection(c)
        self.get(CollectionKeys.HYPERPARAMETERS).include("^hyperparameters/.*$")
        self.get(CollectionKeys.METRICS).include("^[a-zA-z]+-[a-zA-z0-9]+$")
        self.get(CollectionKeys.PREDICTIONS).include("^predictions$")
        self.get(CollectionKeys.LABELS).include("^labels$")
        self.get(CollectionKeys.FEATURE_IMPORTANCE).include("^feature_importance/.*")
        self.get(CollectionKeys.AVERAGE_SHAP).include("^average_shap/.*[^/bias]$")
        self.get(CollectionKeys.FULL_SHAP).include("^full_shap/.*[^/bias]$")
        self.get(CollectionKeys.TREES).include("^trees/.*")
