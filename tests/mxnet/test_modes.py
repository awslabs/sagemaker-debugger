# Standard Library
from datetime import datetime

# First Party
from smdebug import modes
from smdebug.mxnet import SaveConfig, SaveConfigMode, modes
from smdebug.mxnet.hook import Hook as t_hook
from smdebug.trials import create_trial

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_modes(hook=None, path=None):
    if hook is None:
        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        path = "./newlogsRunTest/" + run_id
        hook = t_hook(
            out_dir=path,
            save_config=SaveConfig(
                {
                    modes.TRAIN: SaveConfigMode(save_interval=2),
                    modes.EVAL: SaveConfigMode(save_interval=3),
                }
            ),
            include_collections=["gradients", "weights"],
        )
    run_mnist_gluon_model(
        hook=hook, set_modes=True, register_to_loss_block=True, num_steps_train=6, num_steps_eval=6
    )

    tr = create_trial(path)
    assert len(tr.modes()) == 2
    assert len(tr.steps()) == 5, tr.steps()
    assert len(tr.steps(mode=modes.TRAIN)) == 3
    assert len(tr.steps(mode=modes.EVAL)) == 2, tr.steps()

    # Ensure that the gradients are available in TRAIN modes only.
    grad_tns_name = tr.tensors(regex="^gradient.")[0]
    grad_tns = tr.tensor(grad_tns_name)
    grad_train_steps = grad_tns.steps(mode=modes.TRAIN)
    grad_eval_steps = grad_tns.steps(mode=modes.EVAL)
    assert len(grad_train_steps) == 3
    assert grad_eval_steps == []

    # Ensure that the weights are available in TRAIN and EVAL  modes.
    wt_tns_name = tr.tensors(regex="conv\d+_weight")[0]
    wt_tns = tr.tensor(wt_tns_name)
    wt_train_steps = wt_tns.steps(mode=modes.TRAIN)
    wt_eval_steps = wt_tns.steps(mode=modes.EVAL)
    assert len(wt_train_steps) == 3
    assert len(wt_eval_steps) == 2


def test_modes_hook_from_json_config():
    from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
    import shutil
    import os

    out_dir = "newlogsRunTest2/test_modes_hookjson"
    shutil.rmtree(out_dir, True)
    os.environ[CONFIG_FILE_PATH_ENV_STR] = "tests/mxnet/test_json_configs/test_modes_hook.json"
    hook = t_hook.create_from_json_file()
    test_modes(hook, out_dir)
    shutil.rmtree(out_dir, True)
