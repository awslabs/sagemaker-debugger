# Standard Library
import json
import logging
import os
import shutil
from datetime import datetime

# Third Party
import mxnet as mx
import numpy as np
import pytest
import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost
from mxnet import autograd, gluon, init
from mxnet.gluon import nn as mxnn
from tensorflow import keras
from torch.autograd import Variable

# First Party
from smdebug.core.config_constants import DEFAULT_SAGEMAKER_METRICS_PATH
from smdebug.core.modes import ModeKeys
from smdebug.core.sagemaker_utils import is_sagemaker_job
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.mxnet import Hook as MX_Hook
from smdebug.pytorch import Hook as PT_Hook
from smdebug.tensorflow import KerasHook as TF_KerasHook
from smdebug.tensorflow import SessionHook as TF_SessionHook
from smdebug.trials import create_trial
from smdebug.xgboost import Hook as XG_Hook

SMDEBUG_PT_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_pt/tests/"
SMDEBUG_MX_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_mx/tests/"
SMDEBUG_TF_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_tf/tests/"
SMDEBUG_XG_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_xg/tests/"


def simple_pt_model(hook, steps=10, register_loss=False):
    """
    Create a PT model. save_scalar() calls are inserted before, during and after training.
    Only the scalars with sm_metric=True will be written to a metrics file.
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
            self.add_module("relu0", nn.ReLU())
            self.add_module("max_pool", nn.MaxPool2d(2, stride=2))
            self.add_module("conv2", nn.Conv2d(20, 50, 5, 1))
            self.add_module("relu1", nn.ReLU())
            self.add_module("max_pool2", nn.MaxPool2d(2, stride=2))
            self.add_module("fc1", nn.Linear(4 * 4 * 50, 500))
            self.add_module("relu2", nn.ReLU())
            self.add_module("fc2", nn.Linear(500, 10))

        def forward(self, x):
            x = self.relu0(self.conv1(x))
            x = self.max_pool(x)
            x = self.relu1(self.conv2(x))
            x = self.max_pool2(x)
            x = x.view(-1, 4 * 4 * 50)
            x = self.relu2(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net().to(torch.device("cpu"))
    criterion = nn.NLLLoss()
    hook.register_module(model)
    if register_loss:
        hook.register_loss(criterion)

    hook.save_scalar("pt_num_steps", steps, sm_metric=True)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    hook.save_scalar("pt_before_train", 1, sm_metric=False)
    hook.set_mode(ModeKeys.TRAIN)
    for i in range(steps):
        batch_size = 32
        data, target = torch.rand(batch_size, 1, 28, 28), torch.rand(batch_size).long()
        data, target = data.to(torch.device("cpu")), target.to(torch.device("cpu"))
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad=True))
        if register_loss:
            loss = criterion(output, target)
        else:
            loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    hook.save_scalar("pt_after_train", 1, sm_metric=False)

    model.eval()
    hook.set_mode(ModeKeys.EVAL)
    with torch.no_grad():
        for i in range(steps):
            batch_size = 32
            data, target = torch.rand(batch_size, 1, 28, 28), torch.rand(batch_size).long()
            data, target = data.to("cpu"), target.to("cpu")
            output = model(data)
            if register_loss:
                loss = criterion(output, target)
            else:
                loss = F.nll_loss(output, target)


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


def simple_tf_model(hook, steps=10, lr=0.4):
    """
    Create a TF model. Tensors registered with the SM_METRICS collection will be logged
    to the metrics file.
    """
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    relu_layer = keras.layers.Dense(128, activation="relu")

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            relu_layer,
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    opt = tf.train.RMSPropOptimizer(lr)
    opt = hook.wrap_optimizer(opt)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        run_eagerly=False,
        metrics=["accuracy"],
    )
    hooks = [hook]
    hook.save_scalar("tf_keras_num_steps", steps, sm_metric=True)

    hook.save_scalar("tf_keras_before_train", 1, sm_metric=False)
    hook.set_mode(ModeKeys.TRAIN)
    model.fit(x_train, y_train, epochs=1, steps_per_epoch=steps, callbacks=hooks, verbose=0)

    hook.set_mode(ModeKeys.EVAL)
    model.evaluate(x_test, y_test, steps=10, callbacks=hooks, verbose=0)
    hook.save_scalar("tf_keras_after_train", 1, sm_metric=False)


def tf_session_model(hook, steps=10, lr=0.4):
    # Network definition
    with tf.name_scope("foobar"):
        x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        w = tf.Variable(initial_value=[[10.0], [10.0]], name="weight1")
    with tf.name_scope("foobaz"):
        w0 = [[1], [1.0]]
        y = tf.matmul(x, w0)
    loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")
    hook.get_collection("losses").add(loss)
    global_step = tf.Variable(17, name="global_step", trainable=False)
    increment_global_step_op = tf.assign(global_step, global_step + 1)

    optimizer = tf.train.AdamOptimizer(lr)
    optimizer = hook.wrap_optimizer(optimizer)
    optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

    sess = tf.train.MonitoredSession(hooks=[hook])
    hook.save_scalar("tf_session_num_steps", steps, sm_metric=True)

    hook.save_scalar("tf_session_before_train", 1, sm_metric=False)
    hook.set_mode(ModeKeys.TRAIN)
    for i in range(steps):
        x_ = np.random.random((10, 2)) * 0.1
        _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})

    sess.close()
    hook.save_scalar("tf_session_after_train", 1, sm_metric=False)


def simple_xg_model(hook, num_round=10, seed=42):

    np.random.seed(seed)

    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)

    test_data = np.random.rand(5, 10)
    test_label = np.random.randint(2, size=5)
    dtest = xgboost.DMatrix(test_data, label=test_label)

    params = {}

    hook.save_scalar("xg_num_steps", num_round, sm_metric=True)

    hook.save_scalar("xg_before_train", 1, sm_metric=False)
    hook.set_mode(ModeKeys.TRAIN)
    xgboost.train(
        params,
        dtrain,
        evals=[(dtrain, "train"), (dtest, "test")],
        num_boost_round=num_round,
        callbacks=[hook],
    )
    hook.save_scalar("xg_after_train", 1, sm_metric=False)


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
    if saved_scalars:
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


def helper_pytorch_tests(collection, register_loss, save_config):
    coll_name, coll_regex = collection

    run_id = "trial_" + coll_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(SMDEBUG_PT_HOOK_TESTS_DIR, run_id)

    hook = PT_Hook(
        out_dir=trial_dir,
        include_collections=[coll_name],
        save_config=save_config,
        export_tensorboard=True,
    )

    simple_pt_model(hook, register_loss=register_loss)
    hook.close()

    saved_scalars = ["scalar/pt_num_steps", "scalar/pt_before_train", "scalar/pt_after_train"]
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
def test_pytorch_save_scalar(collection, save_config, register_loss):
    helper_pytorch_tests(collection, register_loss, save_config)
    delete_local_trials([SMDEBUG_PT_HOOK_TESTS_DIR])


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


def helper_tensorflow_tests(use_keras, collection, save_config):
    coll_name, coll_regex = collection

    run_id = "trial_" + coll_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(SMDEBUG_TF_HOOK_TESTS_DIR, run_id)

    if use_keras:
        hook = TF_KerasHook(
            out_dir=trial_dir,
            include_collections=[coll_name],
            save_config=save_config,
            export_tensorboard=True,
        )

        simple_tf_model(hook)

        saved_scalars = [
            "scalar/tf_keras_num_steps",
            "scalar/tf_keras_before_train",
            "scalar/tf_keras_after_train",
        ]
    else:
        hook = TF_SessionHook(
            out_dir=trial_dir,
            include_collections=[coll_name],
            save_config=save_config,
            export_tensorboard=True,
        )

        tf_session_model(hook)
        tf.reset_default_graph()

        saved_scalars = [
            "scalar/tf_session_num_steps",
            "scalar/tf_session_before_train",
            "scalar/tf_session_after_train",
        ]
    hook.close()
    verify_files(trial_dir, save_config, saved_scalars)


@pytest.mark.slow
@pytest.mark.parametrize("use_keras", [True, False])
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
def test_tf_save_scalar(use_keras, collection, save_config):
    helper_tensorflow_tests(use_keras, collection, save_config)
    delete_local_trials([SMDEBUG_TF_HOOK_TESTS_DIR])


def helper_xgboost_tests(collection, save_config):
    coll_name, coll_regex = collection

    run_id = "trial_" + coll_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(SMDEBUG_XG_HOOK_TESTS_DIR, run_id)

    hook = XG_Hook(
        out_dir=trial_dir,
        include_collections=[coll_name],
        save_config=save_config,
        export_tensorboard=True,
    )

    simple_xg_model(hook)
    hook.close()

    saved_scalars = ["scalar/xg_num_steps", "scalar/xg_before_train", "scalar/xg_after_train"]
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
def test_xgboost_save_scalar(collection, save_config):
    helper_xgboost_tests(collection, save_config)
    delete_local_trials([SMDEBUG_XG_HOOK_TESTS_DIR])
