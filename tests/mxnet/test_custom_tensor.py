# Standard Library
import shutil
from datetime import datetime

# First Party
from smdebug import SaveConfig
from smdebug.core.collection import CollectionKeys
from smdebug.mxnet.hook import Hook as t_hook
from smdebug.trials import create_trial

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_hook():
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir = "/tmp/newlogsRunTest/" + run_id
    hook = t_hook(out_dir=out_dir, save_config=save_config)
    run_mnist_gluon_model(
        hook=hook,
        num_steps_train=10,
        num_steps_eval=10,
        register_to_loss_block=True,
        save_custom_tensor=True,
    )
    trial = create_trial(out_dir)
    custom_tensors = trial.tensor_names(collection=CollectionKeys.DEFAULT)
    assert len(custom_tensors)
    shutil.rmtree(out_dir)
