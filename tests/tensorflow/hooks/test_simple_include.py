# Standard Library
import glob

# First Party
from smdebug.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.reader import FileReader
from smdebug.core.utils import get_path_to_collections
from smdebug.tensorflow import SaveConfig, SessionHook
from smdebug.tensorflow.collection import CollectionManager

# Local
from .utils import get_dirs_files, join, os, pre_test_clean_up, simple_model


def helper_test_simple_include(trial_dir, hook):
    hook.get_collection("default").include("loss:0")
    simple_model(hook, steps=10)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))

    cm = CollectionManager.load(
        join(get_path_to_collections(trial_dir), DEFAULT_COLLECTIONS_FILE_NAME)
    )
    assert len(cm.collections["default"].tensor_names) == 1
    assert len(steps) == 5
    for step in steps:
        i = 0
        size = 0
        fs = glob.glob(join(trial_dir, "events", step, "**", "*.tfevents"), recursive=True)
        for f in fs:
            fr = FileReader(f)
            for x in fr.read_tensors():
                tensor_name, step, tensor_data, mode, mode_step = x
                i += 1
                size += tensor_data.nbytes if tensor_data is not None else 0
        assert i == 1
        assert size == 4


def test_simple_include(out_dir):
    pre_test_clean_up()
    hook = SessionHook(
        out_dir=out_dir,
        save_config=SaveConfig(save_interval=2),
        include_collections=["default", "losses"],
    )
    helper_test_simple_include(out_dir, hook)


def test_simple_include_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_simple_include.json",
    )
    hook = SessionHook.create_from_json_file()
    helper_test_simple_include(out_dir, hook)


def helper_test_simple_include_regex(trial_dir, hook):
    simple_model(hook, steps=10)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))

    cm = CollectionManager.load(
        join(get_path_to_collections(trial_dir), DEFAULT_COLLECTIONS_FILE_NAME)
    )
    assert len(cm.collections["default"].tensor_names) == 1
    assert len(steps) == 5

    for step in steps:
        i = 0
        size = 0
        fs = glob.glob(join(trial_dir, "events", step, "**", "*.tfevents"), recursive=True)
        for f in fs:
            fr = FileReader(f)
            for x in fr.read_tensors():
                tensor_name, step, tensor_data, mode, mode_step = x
                i += 1
                size += tensor_data.nbytes if tensor_data is not None else 0
        assert i == 1
        assert size == 4


def test_simple_include_regex(out_dir):
    pre_test_clean_up()
    hook = SessionHook(
        out_dir=out_dir,
        include_regex=["loss:0"],
        include_collections=[],
        save_config=SaveConfig(save_interval=2),
    )
    helper_test_simple_include_regex(out_dir, hook)


def test_simple_include_regex_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_simple_include_regex.json",
    )
    hook = SessionHook.create_from_json_file()
    helper_test_simple_include_regex(out_dir, hook)


def helper_test_multi_collection_match(trial_dir, hook):
    simple_model(hook, steps=10)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))

    cm = CollectionManager.load(
        join(get_path_to_collections(trial_dir), DEFAULT_COLLECTIONS_FILE_NAME)
    )
    assert len(cm.collections["default"].tensor_names) == 1
    assert len(cm.collections["trial"].tensor_names) == 1
    assert len(steps) == 5

    for step in steps:
        i = 0
        size = 0
        fs = glob.glob(join(trial_dir, "events", step, "**", "*.tfevents"), recursive=True)
        for f in fs:
            fr = FileReader(f)
            for x in fr.read_tensors():
                tensor_name, step, tensor_data, mode, mode_step = x
                i += 1
                size += tensor_data.nbytes if tensor_data is not None else 0
        assert i == 1
        assert size == 4


def test_multi_collection_match(out_dir):
    pre_test_clean_up()
    hook = SessionHook(
        out_dir=out_dir,
        include_regex=["loss:0"],
        include_collections=["default", "trial"],
        save_config=SaveConfig(save_interval=2),
    )
    hook.get_collection("trial").include("loss:0")
    helper_test_multi_collection_match(out_dir, hook)


def test_multi_collection_match_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_multi_collection_match.json",
    )
    hook = SessionHook.create_from_json_file()
    hook.get_collection("trial").include("loss:0")
    helper_test_multi_collection_match(out_dir, hook)
