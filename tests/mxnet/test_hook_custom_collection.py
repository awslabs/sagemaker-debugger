# Standard Library
import shutil
from datetime import datetime

# First Party
from smdebug.mxnet import SaveConfig
from smdebug.mxnet.hook import Hook as t_hook

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_hook_custom_collection():
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir = "/tmp/" + run_id
    hook = t_hook(out_dir=out_dir, save_config=save_config, include_collections=["ReluActivation"])
    hook.get_collection("ReluActivation").include(["relu*", "input_*"])
    run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10)
    shutil.rmtree(out_dir)
