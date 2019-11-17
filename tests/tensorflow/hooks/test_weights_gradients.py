# Standard Library

# First Party
import smdebug.tensorflow as smd
from smdebug.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.utils import get_path_to_collections
from smdebug.tensorflow.collection import CollectionManager

# Local
from .utils import get_collection_files, get_dirs_files, join, os, pre_test_clean_up, simple_model


def helper_test_only_w_g(trial_dir, hook):
    simple_model(hook)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))
    collection_files = get_collection_files(trial_dir)

    assert DEFAULT_COLLECTIONS_FILE_NAME in collection_files
    cm = CollectionManager.load(
        join(get_path_to_collections(trial_dir), DEFAULT_COLLECTIONS_FILE_NAME)
    )
    assert hook.get_collections() == cm.collections
    num_tensors_loaded_collection = (
        len(cm.collections["weights"].tensor_names)
        + len(cm.collections["gradients"].tensor_names)
        + len(cm.collections["default"].tensor_names)
    )
    assert num_tensors_loaded_collection == 2
    assert len(steps) == 5


def test_only_w_g(out_dir):
    pre_test_clean_up()
    hook = smd.SessionHook(out_dir, save_all=False, save_config=smd.SaveConfig(save_interval=2))
    helper_test_only_w_g(out_dir, hook)


def test_only_w_g_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_only_weights_and_gradients.json",
    )
    hook = smd.SessionHook.hook_from_config()
    helper_test_only_w_g(out_dir, hook)
