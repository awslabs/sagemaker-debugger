# Standard Library
import os

# First Party
from smdebug.core.collection import Collection
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from smdebug.core.modes import ModeKeys
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.core.utils import get_path_to_collections


def test_export_load():
    # with none as save config
    c1 = Collection(
        "default",
        include_regex=["conv2d"],
        tensor_names=["a", "b"],
        reduction_config=ReductionConfig(),
    )
    c2 = Collection.from_json(c1.to_json())
    assert c1 == c2
    assert c1.tensor_names == c2.tensor_names
    assert isinstance(c2.tensor_names, set)


def test_load_empty():
    c = Collection("trial")
    assert c == Collection.from_json(c.to_json())


def test_export_load_dict_save_config():
    c1 = Collection(
        "default",
        include_regex=["conv2d"],
        reduction_config=ReductionConfig(),
        save_config=SaveConfig(
            {
                ModeKeys.TRAIN: SaveConfigMode(save_interval=10),
                ModeKeys.EVAL: SaveConfigMode(start_step=1),
            }
        ),
    )
    c2 = Collection.from_json(c1.to_json())
    assert c1 == c2
    assert c1.to_json_dict() == c2.to_json_dict()


def test_manager_export_load():
    cm = CollectionManager()
    cm.create_collection("default")
    cm.get("default").include("loss")
    cm.add(Collection("trial1"))
    cm.add("trial2")
    cm.get("trial2").include("total_loss")
    cm.export("/tmp/dummy_trial", DEFAULT_COLLECTIONS_FILE_NAME)
    cm2 = CollectionManager.load(
        os.path.join(get_path_to_collections("/tmp/dummy_trial"), DEFAULT_COLLECTIONS_FILE_NAME)
    )
    assert cm == cm2


def test_manager():
    cm = CollectionManager()
    cm.create_collection("default")
    cm.get("default").include("loss")
    cm.get("default").add_tensor_name("assaas")
    cm.add(Collection("trial1"))
    cm.add("trial2")
    cm.get("trial2").include("total_loss")
    assert len(cm.collections) == 3
    assert cm.get("default") == cm.collections["default"]
    assert "loss" in cm.get("default").include_regex
    assert len(cm.get("default").tensor_names) > 0
    assert "total_loss" in cm.collections["trial2"].include_regex
