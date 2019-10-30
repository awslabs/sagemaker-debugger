from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole.mxnet import SaveConfig, reset_collections
import tornasole.mxnet as tm
import shutil
import os

from datetime import datetime


def test_save_config(hook=None):
    if hook is None:
        reset_collections()
        save_config_collection = SaveConfig(save_steps=[4, 5, 6])

        custom_collect = tm.get_collection("ReluActivation")
        custom_collect.save_config = save_config_collection
        custom_collect.include(["relu*", "input_*", "output*"])
        save_config = SaveConfig(save_steps=[0, 1, 2, 3])
        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        out_dir = "./newlogsRunTest/" + run_id
        hook = t_hook(
            out_dir=out_dir,
            save_config=save_config,
            include_collections=["ReluActivation", "weights", "biases", "gradients", "default"],
        )
    run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10)
    if hook is None:
        shutil.rmtree(out_dir)


def test_save_config_hookjson_config():
    from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR

    reset_collections()
    out_dir = "newlogsRunTest2/test_hook_from_json_config_full"
    shutil.rmtree(out_dir, True)
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_save_config_hookjson_config.json"
    hook = t_hook.hook_from_config()
    test_save_config(hook=hook)
    shutil.rmtree(out_dir, True)
