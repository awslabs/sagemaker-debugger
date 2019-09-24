from tornasole.core.collection_manager import COLLECTIONS_FILE_NAME
from .utils import *
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR

import tornasole.tensorflow as ts
import shutil


def helper_test_when_nan(trial_dir, hook):
    simple_model(hook, steps=100, lr=4e20)
    steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))
    _, files = get_dirs_files(trial_dir)

    assert COLLECTIONS_FILE_NAME in files
    cm = CollectionManager.load(join(trial_dir, COLLECTIONS_FILE_NAME))
    num_tensors_loaded_collection = len(cm.collections['weights'].tensor_names) + \
                                    len(cm.collections['gradients'].tensor_names) + \
                                    len(cm.collections['when_nan'].tensor_names) + \
                                    len(cm.collections['default'].tensor_names)
    assert num_tensors_loaded_collection == 3

    num_steps_with_files = 0
    for step in steps:
        filepath, size = get_event_file_path_length(join(trial_dir, 'events', step))
        if size > 0:
            num_steps_with_files += 1
    assert num_steps_with_files == 35


def test_when_nan():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    pre_test_clean_up()
    hook = TornasoleHook(out_dir=trial_dir,
                         save_config=SaveConfig(save_interval=10, when_nan=['loss:0']))
    helper_test_when_nan(trial_dir, hook)


def test_when_nan_json():
    trial_dir = "newlogsRunTest1/test_when_nan"
    shutil.rmtree(trial_dir, ignore_errors=True)
    pre_test_clean_up()
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = "tests/tensorflow/hooks/test_json_configs/test_when_nan.json"
    hook = ts.TornasoleHook.hook_from_config()
    helper_test_when_nan(trial_dir, hook)
