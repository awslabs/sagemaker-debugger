# Standard Library

# Third Party
import pytest
from tests.tensorflow.hooks.test_estimator_modes import help_test_mnist

# First Party
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.tensorflow import SaveConfig, SaveConfigMode, SessionHook, modes
from smdebug.trials import create_trial

# Local
from .utils import get_collection_files, get_dirs_files, os, pre_test_clean_up, simple_model


def helper_test_save_config(trial_dir, hook):
    simple_model(hook)
    files = get_collection_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))
    assert len(steps) == 5
    assert len(files) == 1


def test_save_config(out_dir):
    pre_test_clean_up()
    hook = SessionHook(out_dir=out_dir, save_all=False, save_config=SaveConfig(save_interval=2))
    helper_test_save_config(out_dir, hook)


def test_save_config_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR, "tests/tensorflow/hooks/test_json_configs/test_save_config.json"
    )
    hook = SessionHook.hook_from_config()
    helper_test_save_config(out_dir, hook)


def helper_save_config_skip_steps(trial_dir, hook):
    simple_model(hook, steps=20)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))
    assert len(steps) == 6


def test_save_config_skip_steps(out_dir):
    pre_test_clean_up()
    hook = SessionHook(
        out_dir=out_dir, save_all=False, save_config=SaveConfig(save_interval=2, start_step=8)
    )
    helper_save_config_skip_steps(out_dir, hook)


def test_save_config_skip_steps_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_save_config_skip_steps.json",
    )
    hook = SessionHook.hook_from_config()
    helper_save_config_skip_steps(out_dir, hook)


def helper_save_config_start_and_end(trial_dir, hook):
    simple_model(hook, steps=20)
    _, files = get_dirs_files(trial_dir)
    steps, _ = get_dirs_files(os.path.join(trial_dir, "events"))
    assert len(steps) == 3


def test_save_config_start_and_end(out_dir):
    pre_test_clean_up()
    hook = SessionHook(
        out_dir=out_dir,
        save_all=False,
        save_config=SaveConfig(save_interval=2, start_step=8, end_step=14),
    )
    helper_save_config_start_and_end(out_dir, hook)


def test_save_config_start_and_end_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_save_config_start_and_end.json",
    )
    hook = SessionHook.hook_from_config()
    helper_save_config_start_and_end(out_dir, hook)


def helper_save_config_modes(trial_dir, hook):
    help_test_mnist(trial_dir, hook=hook, num_steps=2, num_eval_steps=3)
    tr = create_trial(trial_dir)
    for tname in tr.tensors(collection="weights"):
        t = tr.tensor(tname)
        assert len(t.steps(mode=modes.TRAIN)) == 2
        assert len(t.steps(mode=modes.EVAL)) == 1


@pytest.mark.slow  # 0:03 to run
def test_save_config_modes(out_dir):
    pre_test_clean_up()
    hook = SessionHook(out_dir=out_dir, include_collections=["weights"])
    hook.get_collection("weights").save_config = {
        modes.TRAIN: SaveConfigMode(save_interval=2),
        modes.EVAL: SaveConfigMode(save_interval=3),
    }
    helper_save_config_modes(out_dir, hook)


@pytest.mark.slow  # 0:03 to run
def test_save_config_modes_json(out_dir, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_save_config_modes_config_coll.json",
    )
    hook = SessionHook.hook_from_config()
    helper_save_config_modes(out_dir, hook)
