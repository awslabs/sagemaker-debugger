# Standard Library
import json
import logging
import os
import shutil
from datetime import datetime

# Third Party
import mxnet as mx
import pytest
from mxnet import autograd, gluon, init
from mxnet.gluon import nn as mxnn

# First Party
from smdebug.core.config_constants import DEFAULT_SAGEMAKER_METRICS_PATH
from smdebug.core.modes import ModeKeys
from smdebug.core.sagemaker_utils import is_sagemaker_job
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.mxnet import Hook as MX_Hook
from smdebug.trials import create_trial

SMDEBUG_MX_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_mx/tests/"


def simple_mx_model(hook, steps=10, register_loss=False):
    """
    Create a MX model. save_scalar() calls are inserted before, during and after training.
    Only the scalars with sm_metric=True will be written to a metrics file.
    """
    net = mxnn.HybridSequential()
    net.add(
        mxnn.Conv2D(channels=6, kernel_size=5, activation="relu"),
        mxnn.MaxPool2D(pool_size=2, strides=2),
        mxnn.Conv2D(channels=16, kernel_size=3, activation="relu"),
        mxnn.MaxPool2D(pool_size=2, strides=2),
        mxnn.Flatten(),
        mxnn.Dense(120, activation="relu"),
        mxnn.Dense(84, activation="relu"),
        mxnn.Dense(10),
    )
    net.initialize(init=init.Xavier(), ctx=mx.cpu())
    hook.register_block(net)

    train_loss = 0.0
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    if register_loss:
        hook.register_block(softmax_cross_entropy)

    hook.save_scalar("mx_num_steps", steps, sm_metric=True)

    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})

    hook.save_scalar("mx_before_train", 1, sm_metric=False)
    hook.set_mode(ModeKeys.TRAIN)
    for i in range(steps):
        batch_size = 32
        data, target = mx.random.randn(batch_size, 1, 28, 28), mx.random.randn(batch_size)
        data = data.as_in_context(mx.cpu(0))
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, target)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
    hook.save_scalar("mx_after_train", 1, sm_metric=False)

    hook.set_mode(ModeKeys.EVAL)
    for i in range(steps):
        batch_size = 32
        data, target = mx.random.randn(batch_size, 1, 28, 28), mx.random.randn(batch_size)
        data = data.as_in_context(mx.cpu(0))
        val_output = net(data)
        loss = softmax_cross_entropy(val_output, target)


def delete_local_trials(local_trials):
    for trial in local_trials:
        shutil.rmtree(trial)


# need this to seek to the right file offset for test output verification
metrics_file_position = 0


def check_metrics_file(save_steps, saved_scalars=None):
    """
    Check the SageMaker metrics file to ensure that all the scalars saved using
    save_scalar(sm_metrics=True) or mentioned through SM_METRICS collections, have been saved.
    """
    global metrics_file_position
    if is_sagemaker_job():
        METRICS_DIR = os.environ.get(DEFAULT_SAGEMAKER_METRICS_PATH)
        if not METRICS_DIR:
            logging.warning("SageMaker Metric Directory not specified")
            return
        file_name = "{}/{}.json".format(METRICS_DIR, str(os.getpid()))
        scalarnames = set()

        import collections

        train_metric = collections.defaultdict(list)
        eval_metric = collections.defaultdict(list)

        with open(file_name) as fp:
            # since SM metrics expects all metrics to be written in 1 file, seeking to
            # the right offset for the purpose of this test - so that the metrics logged in
            # the corresponding test are verified
            fp.seek(metrics_file_position)
            for line in fp:
                data = json.loads(line)
                assert data["IterationNumber"] != -1  # iteration number should not be -1
                metric_name = data["MetricName"]
                if "TRAIN" in metric_name:
                    train_metric[metric_name].append(data["IterationNumber"])
                    scalarnames.add(metric_name.rstrip("_TRAIN"))
                elif "EVAL" in metric_name:
                    eval_metric[metric_name].append(data["IterationNumber"])
                    scalarnames.add(metric_name.rstrip("_EVAL"))
                else:
                    scalarnames.add(
                        metric_name.rstrip("_GLOBAL")
                    )  # check the scalar saved using save_scalar()
            metrics_file_position = fp.tell()
        assert scalarnames

        if saved_scalars:
            assert len(set(saved_scalars) & set(scalarnames)) > 0

        # check if all metrics have been written at the expected step number
        for train_data in train_metric:
            assert len(set(save_steps["TRAIN"]) & set(train_metric[train_data])) == len(
                save_steps["TRAIN"]
            )
        for eval_data in eval_metric:
            assert len(set(save_steps["EVAL"]) & set(eval_metric[eval_data])) == len(
                save_steps["EVAL"]
            )


def check_trials(out_dir, save_steps, saved_scalars=None):
    """
    Create trial to check if non-scalar data is written as per save config and
    check whether all the scalars written through save_scalar have been saved.
    """
    trial = create_trial(path=out_dir, name="test output")
    assert trial
    tensor_list = trial.tensor_names()
    for tname in tensor_list:
        if tname not in saved_scalars:
            train_steps = trial.tensor(tname).steps(mode=ModeKeys.TRAIN)
            eval_steps = trial.tensor(tname).steps(mode=ModeKeys.EVAL)

            # check if all tensors have been saved according to save steps
            assert len(set(save_steps["TRAIN"]) & set(train_steps)) == len(save_steps["TRAIN"])
            if eval_steps:  # need this check for bias and gradients
                assert len(set(save_steps["EVAL"]) & set(eval_steps)) == len(save_steps["EVAL"])
    scalar_list = trial.tensor_names(regex="^scalar")
    if scalar_list:
        assert len(set(saved_scalars) & set(scalar_list)) == len(saved_scalars)


def verify_files(out_dir, save_config, saved_scalars=None):
    """
    Analyze the tensors saved and verify that metrics are stored correctly in the
    SM metrics json file
    """

    # Retrieve save_step for verification in the trial and the JSON file
    save_config_train_steps = save_config.get_save_config(ModeKeys.TRAIN).save_steps
    if not save_config_train_steps:
        save_interval = save_config.get_save_config(ModeKeys.TRAIN).save_interval
        save_config_train_steps = [i for i in range(0, 10, save_interval)]
    save_config_eval_steps = save_config.get_save_config(ModeKeys.EVAL).save_steps
    if not save_config_eval_steps:
        save_interval = save_config.get_save_config(ModeKeys.EVAL).save_interval
        save_config_eval_steps = [i for i in range(0, 10, save_interval)]

    save_steps = {"TRAIN": save_config_train_steps, "EVAL": save_config_eval_steps}

    check_trials(out_dir, save_steps, saved_scalars)
    check_metrics_file(save_steps, saved_scalars)


def helper_mxnet_tests(collection, register_loss, save_config):
    coll_name, coll_regex = collection

    run_id = "trial_" + coll_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(SMDEBUG_MX_HOOK_TESTS_DIR, run_id)

    hook = MX_Hook(
        out_dir=trial_dir,
        include_collections=[coll_name],
        save_config=save_config,
        export_tensorboard=True,
    )

    simple_mx_model(hook, register_loss=register_loss)
    hook.close()

    saved_scalars = ["scalar/mx_num_steps", "scalar/mx_before_train", "scalar/mx_after_train"]
    verify_files(trial_dir, save_config, saved_scalars)


@pytest.mark.parametrize("collection", [("all", ".*"), ("scalars", "^scalar")])
@pytest.mark.parametrize(
    "save_config",
    [
        SaveConfig(save_steps=[0, 2, 4, 6, 8]),
        SaveConfig(
            {
                ModeKeys.TRAIN: SaveConfigMode(save_interval=2),
                ModeKeys.GLOBAL: SaveConfigMode(save_interval=3),
                ModeKeys.EVAL: SaveConfigMode(save_interval=1),
            }
        ),
    ],
)
@pytest.mark.parametrize("register_loss", [True, False])
def test_mxnet_save_scalar(collection, save_config, register_loss):
    helper_mxnet_tests(collection, register_loss, save_config)
    delete_local_trials([SMDEBUG_MX_HOOK_TESTS_DIR])
