# Standard Library
import os
import shutil
from datetime import datetime

# First Party
from smdebug.mxnet import SaveConfig
from smdebug.mxnet.hook import Hook as t_hook

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_save_config(hook=None):
    if hook is None:
        save_config_collection = SaveConfig(save_steps=[4, 5, 6])
        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        out_dir = "./newlogsRunTest/" + run_id
        save_config = SaveConfig(save_steps=[0, 1, 2, 3])
        hook = t_hook(
            out_dir=out_dir,
            save_config=save_config,
            include_collections=["ReluActivation", "weights", "biases", "gradients", "default"],
        )
        custom_collect = hook.get_collection("ReluActivation")
        custom_collect.save_config = save_config_collection
        custom_collect.include(["relu*", "input_*", "output*"])

    run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10)
    if hook is None:
        shutil.rmtree(out_dir)


def test_save_config_hookjson_config():
    from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR

    out_dir = "newlogsRunTest2/test_hook_from_json_config_full"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_save_config_hookjson_config.json"
    hook = t_hook.hook_from_config()
    test_save_config(hook=hook)
    shutil.rmtree(out_dir, True)
