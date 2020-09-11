# Standard Library
import os
import shutil
from datetime import datetime

# Third Party
import torch
import torch.optim as optim

# First Party
from smdebug.pytorch import ReductionConfig, SaveConfig
from smdebug.pytorch.hook import Hook as t_hook
from smdebug.trials import create_trial

# Local
from .utils import Net, train


def test_reduce_config(hook=None, out_dir=None):
    hook_created = False
    if hook is None:
        global_reduce_config = ReductionConfig(reductions=["max", "mean", "variance"])
        global_save_config = SaveConfig(save_steps=[0, 1, 2, 3])

        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        out_dir = "/tmp/" + run_id
        hook = t_hook(
            out_dir=out_dir,
            save_config=global_save_config,
            include_collections=[
                "weights",
                "biases",
                "gradients",
                "default",
                "ReluActivation",
                "flatten",
            ],
            reduction_config=global_reduce_config,
        )
        hook.get_collection("ReluActivation").include(["relu*"])
        hook.get_collection("ReluActivation").save_config = SaveConfig(save_steps=[4, 5, 6])
        hook.get_collection("ReluActivation").reduction_config = ReductionConfig(
            reductions=["min"], abs_reductions=["max"]
        )
        hook_created = True

    model = Net().to(torch.device("cpu"))
    hook.register_module(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, hook, torch.device("cpu"), optimizer, num_steps=10)

    # Testing
    tr = create_trial(out_dir)
    assert tr
    assert len(tr.steps()) == 7
    print(tr.tensor_names())
    tname = tr.tensor_names(regex="Net_conv[0-9]+.weight")[0]
    print(tr.tensor_names())

    # Global reduction with max and mean and variance
    weight_tensor = tr.tensor(tname)
    max_val = weight_tensor.reduction_value(1, abs=False, reduction_name="max")
    assert max_val != None
    mean_val = weight_tensor.reduction_value(1, abs=False, reduction_name="mean")
    assert mean_val != None
    variance_val = weight_tensor.reduction_value(1, abs=False, reduction_name="variance")
    assert variance_val != None

    # custom reduction at step 4 with reduction = 'min and abs reduction = 'max'
    tname = tr.tensor_names(regex="relu0_input_0")[0]
    relu_input = tr.tensor(tname)
    min_val = relu_input.reduction_value(4, abs=False, reduction_name="min")
    assert min_val != None
    abs_max_val = relu_input.reduction_value(4, abs=True, reduction_name="max")
    assert abs_max_val != None

    # Custom reduction with normalization
    # tname = tr.tensors_matching_regex('flatten._input_0')[0]
    # flatten_input = tr.tensor(tname)
    # l1_norm = flatten_input.reduction_value(step_num=4, abs=False, reduction_name='l1')
    # assert l1_norm != None
    # l2_norm = flatten_input.reduction_value(step_num=4, abs=True, reduction_name='l2')
    # assert l2_norm != None

    if hook_created:
        shutil.rmtree(out_dir)


# Test creating hook by loading the json file with reduction configs.
def test_reduce_config_with_json():
    from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR

    out_dir = "/tmp/test_output/test_hook_reduction_config/jsonloading"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/pytorch/test_json_configs/test_hook_reduction_config.json"
    hook = t_hook.create_from_json_file()
    test_reduce_config(hook=hook, out_dir=out_dir)
    shutil.rmtree(out_dir, True)
