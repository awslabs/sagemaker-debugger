# Standard Library
import datetime

# First Party
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.modes import ModeKeys
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfigMode
from smdebug.exceptions import InvalidCollectionConfiguration
from smdebug.mxnet.hook import Hook


def test_collection_defaults_to_hook_config():
    """Test that hook save_configs propagate to collection defaults.

  For example, if we set ModeKeys.TRAIN: save_interval=10 in the hook
  and ModeKeys.EVAL: save_interval=20 in a collection, we would like the collection to
  be finalized as {ModeKeys.TRAIN: save_interval=10, ModeKeys.EVAL: save_interval=20}.
  """
    cm = CollectionManager()
    cm.create_collection("foo")
    cm.get("foo").include_regex = "*"
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


def test_invalid_collection_config_exception():
    cm = CollectionManager()
    cm.create_collection("foo")

    hook = Hook(
        out_dir="/tmp/test_collections/" + str(datetime.datetime.now()),
        save_config={ModeKeys.TRAIN: SaveConfigMode(save_interval=10)},
        include_collections=["foo"],
        reduction_config=ReductionConfig(save_raw_tensor=True),
    )
    hook.collection_manager = cm
    try:
        hook._prepare_collections()
    except InvalidCollectionConfiguration:
        pass
    else:
        assert False, "Invalid Collection Name did not raise error"

    cm.get("foo").include_regex = "*"
    try:
        hook._prepare_collections()
    except InvalidCollectionConfiguration:
        assert False, "Valid Collection Name raised an error"
