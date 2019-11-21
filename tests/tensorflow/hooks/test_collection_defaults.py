# Standard Library

# First Party
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.modes import ModeKeys
from smdebug.tensorflow.session import SessionHook

# Local
from .utils import pre_test_clean_up


def test_collection_defaults_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_collection_defaults.json",
    )
    hook = SessionHook.hook_from_config()

    # Check save_intervals for each mode
    assert hook.save_config.get_save_config(ModeKeys.TRAIN).save_interval == 2
    assert hook.save_config.get_save_config(ModeKeys.EVAL).save_interval == 3
    assert hook.save_config.get_save_config(ModeKeys.PREDICT).save_interval == 1
    assert hook.save_config.get_save_config(ModeKeys.GLOBAL).save_interval == 1
    # Check include_collections
    assert "weights" in hook.include_collections and "losses" in hook.include_collections

    assert len(hook.include_collections) == 4
    # Check collection configurations for losses
    assert (
        hook.collection_manager.collections["losses"]
        .save_config.get_save_config(ModeKeys.TRAIN)
        .save_interval
        == 2
    )
    assert (
        hook.collection_manager.collections["losses"]
        .save_config.get_save_config(ModeKeys.EVAL)
        .save_interval
        == 4
    )
    assert (
        hook.collection_manager.collections["losses"]
        .save_config.get_save_config(ModeKeys.PREDICT)
        .save_interval
        == 1
    )
    assert (
        hook.collection_manager.collections["losses"]
        .save_config.get_save_config(ModeKeys.GLOBAL)
        .save_interval
        == 5
    )
    # Check collection configuration for weights
    assert (
        hook.collection_manager.collections["weights"]
        .save_config.get_save_config(ModeKeys.TRAIN)
        .save_interval
        == 2
    )
    assert (
        hook.collection_manager.collections["weights"]
        .save_config.get_save_config(ModeKeys.EVAL)
        .save_interval
        == 3
    )
    assert (
        hook.collection_manager.collections["weights"]
        .save_config.get_save_config(ModeKeys.PREDICT)
        .save_interval
        == 1
    )
    assert (
        hook.collection_manager.collections["weights"]
        .save_config.get_save_config(ModeKeys.GLOBAL)
        .save_interval
        == 1
    )
