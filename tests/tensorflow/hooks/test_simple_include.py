from .utils import *
from tornasole.tensorflow import get_collection
import tornasole.tensorflow as ts
import glob, shutil
from tornasole.core.reader import FileReader
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
from tornasole.core.config_constants import TORASOLE_DEFAULT_COLLECTIONS_FILE_NAME



def helper_test_simple_include(trial_dir, hook):
    get_collection('default').include('loss:0')
    simple_model(hook, steps=10)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))

    cm = CollectionManager.load(join(trial_dir, TORASOLE_DEFAULT_COLLECTIONS_FILE_NAME))
    assert len(cm.collections['default'].tensor_names) == 1
    assert len(steps) == 5
    for step in steps:
        i = 0
        size = 0
        fs = glob.glob(join(trial_dir, 'events', step, '**', '*.tfevents'),
                       recursive=True)
        for f in fs:
            fr = FileReader(f)
            for x in fr.read_tensors():
                tensor_name, step, tensor_data, mode, mode_step = x
                i += 1
                size += tensor_data.nbytes if tensor_data is not None else 0
        assert i == 3
        assert size == 20

    shutil.rmtree(trial_dir)


def test_simple_include():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    pre_test_clean_up()
    hook = TornasoleHook(out_dir=trial_dir,
                         save_config=SaveConfig(save_interval=2))
    helper_test_simple_include(trial_dir, hook)


def test_simple_include_json():
    trial_dir = "newlogsRunTest1/test_simple_include_json"
    shutil.rmtree(trial_dir, ignore_errors=True)
    pre_test_clean_up()
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = "tests/tensorflow/hooks/test_json_configs/test_simple_include.json"
    hook = TornasoleHook.hook_from_config()
    helper_test_simple_include(trial_dir, hook)


def helper_test_simple_include_regex(trial_dir, hook):
    simple_model(hook, steps=10)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))

    cm = CollectionManager.load(join(trial_dir, TORASOLE_DEFAULT_COLLECTIONS_FILE_NAME))
    assert len(cm.collections['default'].tensor_names) == 1
    assert len(steps) == 5

    for step in steps:
        i = 0
        size = 0
        fs = glob.glob(join(trial_dir, 'events', step, '**', '*.tfevents'),
                       recursive=True)
        for f in fs:
            fr = FileReader(f)
            for x in fr.read_tensors():
                tensor_name, step, tensor_data, mode, mode_step = x
                i += 1
                size += tensor_data.nbytes if tensor_data is not None else 0
        assert i == 1
        assert size == 4

    shutil.rmtree(trial_dir)


def test_simple_include_regex():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    pre_test_clean_up()
    hook = TornasoleHook(out_dir=trial_dir,
                         include_regex=['loss:0'],
                         include_collections=[],
                         save_config=SaveConfig(save_interval=2))
    helper_test_simple_include_regex(trial_dir, hook)


def test_simple_include_regex_json():
    trial_dir = "newlogsRunTest1/test_simple_include_regex_json"
    shutil.rmtree(trial_dir, ignore_errors=True)
    pre_test_clean_up()
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = "tests/tensorflow/hooks/test_json_configs/test_simple_include_regex.json"
    hook = TornasoleHook.hook_from_config()
    helper_test_simple_include_regex(trial_dir, hook)


def helper_test_multi_collection_match(trial_dir, hook):
    simple_model(hook, steps=10)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))

    cm = CollectionManager.load(join(trial_dir, TORASOLE_DEFAULT_COLLECTIONS_FILE_NAME))
    assert len(cm.collections['default'].tensor_names) == 1
    assert len(cm.collections['trial'].tensor_names) == 1
    assert len(steps) == 5

    for step in steps:
        i = 0
        size = 0
        fs = glob.glob(join(trial_dir, 'events', step, '**', '*.tfevents'),
                       recursive=True)
        for f in fs:
            fr = FileReader(f)
            for x in fr.read_tensors():
                tensor_name, step, tensor_data, mode, mode_step = x
                i += 1
                size += tensor_data.nbytes if tensor_data is not None else 0
        assert i == 1
        assert size == 4
    shutil.rmtree(trial_dir)


def test_multi_collection_match():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    pre_test_clean_up()
    ts.get_collection('trial').include('loss:0')
    hook = TornasoleHook(out_dir=trial_dir,
                         include_regex=['loss:0'],
                         include_collections=['default', 'trial'],
                         save_config=SaveConfig(save_interval=2))
    helper_test_multi_collection_match(trial_dir, hook)


def test_multi_collection_match_json():
    trial_dir = "newlogsRunTest1/test_multi_collection_match_json"
    shutil.rmtree(trial_dir, ignore_errors=True)
    pre_test_clean_up()
    ts.get_collection('trial').include('loss:0')
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = "tests/tensorflow/hooks/test_json_configs/test_multi_collection_match.json"
    hook = TornasoleHook.hook_from_config()
    helper_test_multi_collection_match(trial_dir, hook)
