# Standard Library
import os
import time
from datetime import datetime

# Third Party
import mxnet as mx
import pytest
from mxnet import autograd, gluon, init
from mxnet.gluon import nn as mxnn
from tests.core.utils import check_tf_events, delete_local_trials, verify_files

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.mxnet import Hook as MX_Hook

SMDEBUG_MX_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_mx/tests/"


def simple_mx_model(hook, steps=10, register_loss=False, with_timestamp=False):
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

    scalars_to_be_saved = dict()
    ts = time.time()
    hook.save_scalar(
        "mx_num_steps", steps, sm_metric=True, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/mx_num_steps"] = (ts, steps)

    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})

    ts = time.time()
    hook.save_scalar(
        "mx_before_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/mx_before_train"] = (ts, 1)

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

    ts = time.time()
    hook.save_scalar("mx_after_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None)
    scalars_to_be_saved["scalar/mx_after_train"] = (ts, 1)

    hook.set_mode(ModeKeys.EVAL)
    for i in range(steps):
        batch_size = 32
        data, target = mx.random.randn(batch_size, 1, 28, 28), mx.random.randn(batch_size)
        data = data.as_in_context(mx.cpu(0))
        val_output = net(data)
        loss = softmax_cross_entropy(val_output, target)
    return scalars_to_be_saved


def helper_mxnet_tests(collection, register_loss, save_config, with_timestamp):
    coll_name, coll_regex = collection

    run_id = "trial_" + coll_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(SMDEBUG_MX_HOOK_TESTS_DIR, run_id)

    hook = MX_Hook(
        out_dir=trial_dir,
        include_collections=[coll_name],
        save_config=save_config,
        export_tensorboard=True,
    )

    saved_scalars = simple_mx_model(
        hook, register_loss=register_loss, with_timestamp=with_timestamp
    )
    hook.close()

    verify_files(trial_dir, save_config, saved_scalars)
    if with_timestamp:
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
@pytest.mark.parametrize("with_timestamp", [True, False])
def test_mxnet_save_scalar(collection, save_config, register_loss, with_timestamp):
    helper_mxnet_tests(collection, register_loss, save_config, with_timestamp)
    delete_local_trials([SMDEBUG_MX_HOOK_TESTS_DIR])
