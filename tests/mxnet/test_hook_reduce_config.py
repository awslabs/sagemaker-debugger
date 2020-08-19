# Standard Library
import shutil
from datetime import datetime

# Third Party
from tests.utils import verify_shapes

# First Party
from smdebug.mxnet import ReductionConfig, SaveConfig
from smdebug.mxnet.hook import Hook as t_hook
from smdebug.trials import create_trial

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_save_config(hook=None, out_dir=None):
    hook_created = False
    if hook is None:
        hook_created = True
        global_reduce_config = ReductionConfig(reductions=["max", "mean"])
        global_save_config = SaveConfig(save_steps=[0, 1, 2, 3])

        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        out_dir = "/tmp/newlogsRunTest/" + run_id
        print("Registering the hook with out_dir {0}".format(out_dir))
        hook = t_hook(
            out_dir=out_dir,
            save_config=global_save_config,
            save_all=True,
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

        hook.get_collection("flatten").include(["flatten*"])
        hook.get_collection("flatten").save_config = SaveConfig(save_steps=[4, 5, 6])
        hook.get_collection("flatten").reduction_config = ReductionConfig(
            norms=["l1"], abs_norms=["l2"]
        )

    run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10)

    # Testing
    print("Created the trial with out_dir {0}".format(out_dir))
    tr = create_trial(out_dir)
    assert tr
    assert len(tr.steps()) == 7

    print(tr.tensor_names())
    tname = tr.tensor_names(regex=r"conv\d+_weight")[0]
    # Global reduction with max and mean
    weight_tensor = tr.tensor(tname)
    max_val = weight_tensor.reduction_value(step_num=1, abs=False, reduction_name="max")
    assert max_val is not None
    mean_val = weight_tensor.reduction_value(step_num=1, abs=False, reduction_name="mean")
    assert mean_val is not None

    # custom reduction at step 4 with reduction = 'min' and abs reduction = 'max'
    tname = tr.tensor_names(regex=r"conv\d+_relu_input_0")[0]
    relu_input = tr.tensor(tname)
    min_val = relu_input.reduction_value(step_num=4, abs=False, reduction_name="min")
    assert min_val is not None
    abs_max_val = relu_input.reduction_value(step_num=4, abs=True, reduction_name="max")
    assert abs_max_val is not None

    # Custom reduction with normalization
    tname = tr.tensor_names(regex=r"flatten\d+_input_0")[0]
    flatten_input = tr.tensor(tname)
    l1_norm = flatten_input.reduction_value(step_num=4, abs=False, reduction_name="l1")
    assert l1_norm is not None
    l2_norm = flatten_input.reduction_value(step_num=4, abs=True, reduction_name="l2")
    assert l2_norm is not None
    if hook_created:
        shutil.rmtree(out_dir)


def test_save_shapes(out_dir):
    global_reduce_config = ReductionConfig(save_shape=True)
    global_save_config = SaveConfig(save_steps=[0, 1])

    hook = t_hook(
        out_dir=out_dir,
        save_config=global_save_config,
        save_all=True,
        reduction_config=global_reduce_config,
    )
    run_mnist_gluon_model(hook=hook, num_steps_train=5)
    verify_shapes(out_dir, 0)
    verify_shapes(out_dir, 1)
    shutil.rmtree(out_dir)


def test_save_config_hook_from_json():
    from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
    import os

    out_dir = "/tmp/newlogsRunTest2/test_hook_reduce_config_hook_from_json"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_hook_reduce_config_hook.json"
    hook = t_hook.create_from_json_file()
    test_save_config(hook, out_dir)
    # delete output
    shutil.rmtree(out_dir, True)
