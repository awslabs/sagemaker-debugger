# Standard Library
import datetime
import os

# First Party
from smdebug.core.collection import Collection
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from smdebug.core.modes import ModeKeys
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.core.utils import get_path_to_collections
from smdebug.mxnet.hook import Hook


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
    cm.export("dummy_trial", DEFAULT_COLLECTIONS_FILE_NAME)
    cm2 = CollectionManager.load(
        os.path.join(get_path_to_collections("dummy_trial"), DEFAULT_COLLECTIONS_FILE_NAME)
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


def test_collection_defaults_to_hook_config():
    """Test that hook save_configs propagate to collection defaults.

  For example, if we set ModeKeys.TRAIN: save_interval=10 in the hook
  and ModeKeys.EVAL: save_interval=20 in a collection, we would like the collection to
  be finalized as {ModeKeys.TRAIN: save_interval=10, ModeKeys.EVAL: save_interval=20}.
  """
    cm = CollectionManager()
    cm.create_collection("foo")
    cm.get("foo").save_config = {ModeKeys.EVAL: SaveConfigMode(save_interval=20)}

    hook = Hook(
        out_dir="/tmp/test_collections/" + str(datetime.datetime.now()),
        save_config={ModeKeys.TRAIN: SaveConfigMode(save_interval=10)},
        include_collections=["foo"],
        reduction_config=ReductionConfig(save_raw_tensor=True),
    )
    hook.collection_manager = cm
    assert cm.get("foo").save_config.mode_save_configs[ModeKeys.TRAIN] is None
    assert cm.get("foo").reduction_config is None
    hook._prepare_collections()
    assert cm.get("foo").save_config.mode_save_configs[ModeKeys.TRAIN].save_interval == 10
    assert cm.get("foo").reduction_config.save_raw_tensor is True
