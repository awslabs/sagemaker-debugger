# Standard Library
import shutil
from datetime import datetime

# First Party
from smdebug.mxnet import SaveConfig
from smdebug.mxnet.hook import Hook as t_hook
from smdebug.trials import create_trial

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_save_all(hook=None, out_dir=None):
    hook_created = False
    if hook is None:
        hook_created = True
        save_config = SaveConfig(save_steps=[0, 1, 2, 3])
        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        out_dir = "./newlogsRunTest/" + run_id
        print("Registering the hook with out_dir {}".format(out_dir))
        hook = t_hook(out_dir=out_dir, save_config=save_config, save_all=True)
    run_mnist_gluon_model(hook=hook, num_steps_train=7, num_steps_eval=5)
    # assert for steps and tensor_names
    print("Created the trial with out_dir {}".format(out_dir))
    tr = create_trial(out_dir)
    tensor_list = tr.tensor_names()
    assert tr
    assert len(tr.steps()) == 4
    # some tensor names, like input and output, can't be retrieved from training session, so here we only assert for tensor numbers
    # 46 is gotten from index file
    # if no assertion failure, then the script could save all tensors
    assert len(tensor_list) == 46
    if hook_created:
        shutil.rmtree(out_dir)


def test_save_all_hook_from_json():
    from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
    import os

    out_dir = "newlogsRunTest2/test_hook_save_all_hook_from_json"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_hook_save_all_hook.json"
    hook = t_hook.create_from_json_file()
    test_save_all(hook, out_dir)
    # delete output
    shutil.rmtree(out_dir, True)
