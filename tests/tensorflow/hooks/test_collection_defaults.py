# Standard Library
from tempfile import TemporaryDirectory

# First Party
from smdebug.core.collection import CollectionKeys
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.modes import ModeKeys
from smdebug.tensorflow import SaveConfig
from smdebug.tensorflow.session import SessionHook

# Local
from .utils import pre_test_clean_up


def test_collection_defaults_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_collection_defaults.json",
    )
    hook = SessionHook.create_from_json_file()

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


def test_get_custom_and_default_collections():
    tmp_dir = TemporaryDirectory().name

    include_collections = [
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.LOSSES,
        CollectionKeys.OUTPUTS,
        CollectionKeys.METRICS,
        CollectionKeys.LOSSES,
        CollectionKeys.OPTIMIZER_VARIABLES,
        "custom_collection",
    ]

    hook = SessionHook(
        out_dir=tmp_dir,
        save_config=SaveConfig(save_interval=2),
        include_collections=include_collections,
    )
    hook.get_collection(name="custom_collection").include("random-regex")

    custom_collections, default_collections = hook._get_custom_and_default_collections()

    assert len(custom_collections) == 1
    assert (
        len(default_collections) == 8 + 3
    )  # Addtional three collections are: all, default and sm_metrics
