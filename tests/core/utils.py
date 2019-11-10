from tornasole.core.collection import Collection
from tornasole.core.collection_manager import CollectionManager
from tornasole.core.config_constants import TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME


def write_dummy_collection_file(trial):
    cm = CollectionManager()
    cm.create_collection("default")
    cm.add(Collection(trial))
    cm.export(trial, TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME)
