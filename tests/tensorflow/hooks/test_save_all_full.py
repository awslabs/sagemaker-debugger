# Standard Library

# Third Party
import tensorflow as tf
from tests.tensorflow.utils import create_trial_fast_refresh

# First Party
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.tensorflow import SaveConfig, SessionHook

# Local
from .utils import simple_model


def test_save_all_full(out_dir, hook=None):
    tf.reset_default_graph()
    if hook is None:
        hook = SessionHook(
            out_dir=out_dir,
            save_all=True,
            include_collections=["weights", "gradients"],
            save_config=SaveConfig(save_interval=2),
        )

    simple_model(hook)
    tr = create_trial_fast_refresh(out_dir)
    # assert len(tr.tensors()) > 50
    print(tr.tensors(collection="weights"))
    assert len(tr.tensors(collection="weights")) == 1
    assert len(tr.tensors(collection="gradients")) == 1
    # assert len(tr.tensors(collection="losses")) == 1


def test_hook_config_json(out_dir, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_hook_from_json_config.json",
    )
    hook = SessionHook.hook_from_config()
    test_save_all_full(out_dir, hook)
