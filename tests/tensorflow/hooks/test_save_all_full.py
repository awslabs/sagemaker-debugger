# Standard Library

# Third Party
from tests.tensorflow.utils import create_trial_fast_refresh

# First Party
from smdebug.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.utils import get_path_to_collections
from smdebug.tensorflow import CollectionManager, get_collections, reset_collections

# Local
from .utils import SaveConfig, SessionHook, get_collection_files, join, simple_model, tf


def test_save_all_full(out_dir, hook=None):
    tf.reset_default_graph()
    if hook is None:
        reset_collections()
        hook = SessionHook(out_dir=out_dir, save_all=True, save_config=SaveConfig(save_interval=2))

    simple_model(hook)
    files = get_collection_files(out_dir)

    coll = get_collections()
    assert all(
        [x in coll.keys() for x in ["all", "weights", "gradients", "losses", "optimizer_variables"]]
    )
    assert len(coll["weights"].tensor_names) == 1
    assert len(coll["gradients"].tensor_names) == 1
    assert len(coll["losses"].tensor_names) == 1

    assert DEFAULT_COLLECTIONS_FILE_NAME in files
    cm = CollectionManager.load(
        join(get_path_to_collections(out_dir), DEFAULT_COLLECTIONS_FILE_NAME)
    )

    assert len(cm.collections) == len(coll), (coll, cm.collections)
    assert len(cm.collections["weights"].tensor_names) == 1
    assert len(cm.collections["losses"].tensor_names) == 1
    assert len(cm.collections["gradients"].tensor_names) == 1
    # as we hadn't asked to be saved
    assert len(cm.collections["optimizer_variables"].tensor_names) == 0
    assert len(cm.collections["all"].tensor_names) == 106
    num_tensors_loaded_collection = len(cm.collections["weights"].tensor_names) + len(
        cm.collections["gradients"].tensor_names
    )
    num_tensors_collection = len(coll["weights"].tensor_names) + len(coll["gradients"].tensor_names)
    assert num_tensors_collection == num_tensors_loaded_collection

    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensors()) == 101


def test_hook_config_json(out_dir, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_hook_from_json_config.json",
    )
    reset_collections()
    hook = SessionHook.hook_from_config()
    test_save_all_full(out_dir, hook)
