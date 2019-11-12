# Standard Library
import os
import shutil
from datetime import datetime

# Third Party
import pytest

# First Party
from smdebug import SaveConfig
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR, DEFAULT_SAGEMAKER_OUTDIR
from smdebug.mxnet import reset_collections
from smdebug.mxnet.hook import TornasoleHook as t_hook

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_hook():
    reset_collections()
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir = "newlogsRunTest/" + run_id
    hook = t_hook(out_dir=out_dir, save_config=save_config)
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )
    shutil.rmtree(out_dir)


def test_hook_from_json_config():
    reset_collections()
    out_dir = "newlogsRunTest1/test_hook_from_json_config"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_hook_from_json_config.json"
    hook = t_hook.hook_from_config()
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )
    shutil.rmtree(out_dir, True)


def test_hook_from_json_config_full():
    reset_collections()
    out_dir = "newlogsRunTest2/test_hook_from_json_config_full"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_hook_from_json_config_full.json"
    hook = t_hook.hook_from_config()
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )
    shutil.rmtree(out_dir, True)


@pytest.mark.skip(reason="If no config file is found, then SM doesn't want a TornasoleHook")
def test_default_hook():
    reset_collections()
    shutil.rmtree("/opt/ml/output/tensors", ignore_errors=True)
    if CONFIG_FILE_PATH_ENV_STR in os.environ:
        del os.environ[CONFIG_FILE_PATH_ENV_STR]
    hook = t_hook.hook_from_config()
    assert hook.out_dir == DEFAULT_SAGEMAKER_OUTDIR
