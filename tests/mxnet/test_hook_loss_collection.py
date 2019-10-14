from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole import SaveConfig
from tornasole.mxnet import reset_collections
from datetime import datetime
import shutil
from tornasole.core.access_layer.utils import has_training_ended
from tornasole.trials import create_trial
import os


def test_loss_collection_default():
    reset_collections()
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir = "newlogsRunTest/" + run_id
    hook = t_hook(out_dir=out_dir, save_config=save_config)
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )

    print("Created the trial with out_dir {0}".format(out_dir))
    tr = create_trial(out_dir)
    assert tr
    assert len(tr.available_steps()) == 4

    print(tr.tensors())
    tname = tr.tensors_matching_regex(".*loss")[0]
    loss_tensor = tr.tensor(tname)
    loss_val = loss_tensor.value(step_num=1)
    assert len(loss_val) > 0

    shutil.rmtree(out_dir)


def test_loss_collection_with_no_other_collections():
    reset_collections()
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir = "newlogsRunTest/" + run_id
    hook = t_hook(out_dir=out_dir, save_config=save_config, include_collections=[])
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )

    print("Created the trial with out_dir {0}".format(out_dir))
    tr = create_trial(out_dir)
    assert tr
    assert len(tr.available_steps()) == 4

    print(tr.tensors())
    tname = tr.tensors_matching_regex(".*loss")[0]
    loss_tensor = tr.tensor(tname)
    loss_val = loss_tensor.value(step_num=1)
    assert len(loss_val) > 0

    shutil.rmtree(out_dir)
