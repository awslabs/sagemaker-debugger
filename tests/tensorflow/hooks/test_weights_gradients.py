from .utils import *
from tornasole.tensorflow import reset_collections
import tensorflow as tf
from tornasole.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from tornasole.core.json_config import CONFIG_FILE_PATH_ENV_STR
from tornasole.core.utils import get_path_to_collections
import tornasole.tensorflow as ts
import shutil


def helper_test_only_w_g(trial_dir, hook):
    simple_model(hook)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))
    collection_files = get_collection_files(trial_dir)

    assert DEFAULT_COLLECTIONS_FILE_NAME in collection_files
    cm = CollectionManager.load(
        join(get_path_to_collections(trial_dir), DEFAULT_COLLECTIONS_FILE_NAME)
    )
    assert ts.get_collections() == cm.collections
    num_tensors_loaded_collection = (
        len(cm.collections["weights"].tensor_names)
        + len(cm.collections["gradients"].tensor_names)
        + len(cm.collections["default"].tensor_names)
    )
    assert num_tensors_loaded_collection == 2
    assert len(steps) == 5
    # for step in steps:
    #   i = 0
    #   filepath, size = get_event_file_path_length(join(rank_dir, step))
    #   for (n, t) in get_tensors_from_event_file(filepath):
    #     i += 1
    #   assert i == 2


def test_only_w_g():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    pre_test_clean_up()
    hook = TornasoleHook(out_dir=trial_dir, save_all=False, save_config=SaveConfig(save_interval=2))
    helper_test_only_w_g(trial_dir, hook)


def test_only_w_g_json():
    trial_dir = "newlogsRunTest1/test_only_weights_and_gradients"
    shutil.rmtree(trial_dir, ignore_errors=True)
    pre_test_clean_up()
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/tensorflow/hooks/test_json_configs/test_only_weights_and_gradients.json"
    hook = ts.TornasoleHook.hook_from_config()
    helper_test_only_w_g(trial_dir, hook)
