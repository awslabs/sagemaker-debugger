# Standard Library
import glob
import os
import shutil
import time
from datetime import datetime
from os.path import join

# Third Party
import mxnet as mx
import pytest
from mxnet import autograd, gluon, init
from mxnet.gluon import nn as mxnn

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.reader import FileReader
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.mxnet import Hook as MX_Hook

SMDEBUG_PT_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_pt/tests/"
SMDEBUG_MX_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_mx/tests/"
SMDEBUG_TF_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_tf/tests/"
SMDEBUG_XG_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_xg/tests/"


def simple_mx_model(hook, steps=10, register_loss=False):
    """
    Create a MX model. save_scalar() calls are inserted before, during and after training.
    Only the scalars with sm_metric=True will be written to a metrics file.
    The function returns a dictionary of scalar_name to (timestamp, scalar_value) tuple.
    The timestamp stores the time at which the script will invoke save_scalar() method.
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

    # Dictionary to store the scalar, timestamp and value to be saved.
    scalars_saved = dict()

    num_steps_timestamp = time.time()
    hook.save_scalar("mx_num_steps", steps, sm_metric=True, timestamp=num_steps_timestamp)
    scalars_saved["scalar/mx_num_steps"] = (num_steps_timestamp, steps)

    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})

    before_timestamp = time.time()
    hook.save_scalar("mx_before_train", 1234, sm_metric=True, timestamp=before_timestamp)
    scalars_saved["scalar/mx_before_train"] = (before_timestamp, 1234)

    for i in range(steps):
        batch_size = 32
        data, target = mx.random.randn(batch_size, 1, 28, 28), mx.random.randn(batch_size)
        data = data.as_in_context(mx.cpu(0))
        hook.save_scalar("step_start", 1, sm_metric=True)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, target)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        time.sleep(1.0)
        hook.save_scalar("step_start", 0, sm_metric=True)
        # calculate training metrics
        train_loss += loss.mean().asscalar()

    after_timestamp = time.time()
    hook.save_scalar("mx_after_train", 4321, sm_metric=True, timestamp=after_timestamp)
    scalars_saved["scalar/mx_after_train"] = (after_timestamp, 4321)

    return scalars_saved


def delete_local_trials(local_trials):
    for trial in local_trials:
        shutil.rmtree(trial)


"""
    Read the scalar events from tfevents files.
    Test and assert on following:
    1. The names of scalars in 'saved_scalars' match with the names in tfevents.
    2. The timestamps along with the 'saved_scalars' match with timestamps saved in tfevents
    3. The values of 'saved_scalars' match with the values saved in tfevents.
"""


def check_tf_events(out_dir, saved_scalars=None):
    # Read the events from all the saved steps
    fs = glob.glob(join(out_dir, "events", "*", "*.tfevents"), recursive=True)
    events = list()
    for f in fs:
        fr = FileReader(f)
        events += fr.read_events(regex_list=["scalar"])

    # Create a dict of scalar events.
    scalar_events = dict()
    for x in events:
        event_name = str(x["name"])
        if event_name not in scalar_events:
            scalar_events[event_name] = list()
        scalar_events[event_name].append((x["timestamp"], x["value"]))

    for scalar_name in saved_scalars:
        assert scalar_name in scalar_events
        (stored_timestamp, stored_value) = scalar_events[scalar_name][0]
        (recorded_timestamp, recorded_value) = saved_scalars[scalar_name]
        assert recorded_timestamp == stored_timestamp
        assert recorded_value == stored_value[0]


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

    saved_scalars = simple_mx_model(hook, register_loss=register_loss)
    hook.close()

    check_tf_events(trial_dir, saved_scalars)


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
