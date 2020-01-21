# Standard Library

# Third Party
import tensorflow.compat.v1 as tf
from tests.tensorflow.utils import create_trial_fast_refresh

# First Party
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.tensorflow import SaveConfig, SessionHook

# Local
from .utils import simple_model


def test_save_all_full(out_dir, hook=None):
    tf.reset_default_graph()
    if hook is None:
        hook = SessionHook(out_dir=out_dir, save_all=True, save_config=SaveConfig(save_interval=2))

    simple_model(hook)
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensor_names()) > 50
    print(tr.tensor_names(collection="weights"))
    assert len(tr.tensor_names(collection="weights")) == 1
    assert len(tr.tensor_names(collection="gradients")) == 1
    assert len(tr.tensor_names(collection="losses")) == 1


def test_hook_config_json(out_dir, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_hook_from_json_config.json",
    )
    hook = SessionHook.create_from_json_file()
    test_save_all_full(out_dir, hook)
