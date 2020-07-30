# Standard Library
import os
import time
from datetime import datetime

# Third Party
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from tests.core.utils import check_tf_events, delete_local_trials, verify_files

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.tensorflow import KerasHook as TF_KerasHook
from smdebug.tensorflow import SessionHook as TF_SessionHook

SMDEBUG_TF_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_tf/tests/"


def simple_tf_model(hook, steps=10, lr=0.4, with_timestamp=False):
    """
    Create a TF model. Tensors registered with the SM_METRICS collection will be logged
    to the metrics file.
    """
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data(TEST_DATASET_S3_PATH)
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
    scalars_to_be_saved = dict()

    ts = time.time()
    hook.save_scalar(
        "tf_keras_num_steps", steps, sm_metric=True, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/tf_keras_num_steps"] = (ts, steps)

    ts = time.time()
    hook.save_scalar(
        "tf_keras_before_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/tf_keras_before_train"] = (ts, 1)

    hook.set_mode(ModeKeys.TRAIN)
    model.fit(x_train, y_train, epochs=1, steps_per_epoch=steps, callbacks=hooks, verbose=0)

    hook.set_mode(ModeKeys.EVAL)
    model.evaluate(x_test, y_test, steps=10, callbacks=hooks, verbose=0)

    ts = time.time()
    hook.save_scalar(
        "tf_keras_after_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/tf_keras_after_train"] = (ts, 1)
    return scalars_to_be_saved


def tf_session_model(hook, steps=10, lr=0.4, with_timestamp=False):
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

    scalars_to_be_saved = dict()
    ts = time.time()
    hook.save_scalar(
        "tf_session_num_steps", steps, sm_metric=True, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/tf_session_num_steps"] = (ts, steps)

    ts = time.time()
    hook.save_scalar(
        "tf_session_before_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/tf_session_before_train"] = (ts, 1)
    hook.set_mode(ModeKeys.TRAIN)
    for i in range(steps):
        x_ = np.random.random((10, 2)) * 0.1
        _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})

    sess.close()
    ts = time.time()
    hook.save_scalar(
        "tf_session_after_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/tf_session_after_train"] = (ts, 1)
    return scalars_to_be_saved


def helper_tensorflow_tests(use_keras, collection, save_config, with_timestamp):
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

        saved_scalars = simple_tf_model(hook, with_timestamp=with_timestamp)

    else:
        hook = TF_SessionHook(
            out_dir=trial_dir,
            include_collections=[coll_name],
            save_config=save_config,
            export_tensorboard=True,
        )

        saved_scalars = tf_session_model(hook, with_timestamp=with_timestamp)
        tf.reset_default_graph()

    hook.close()
    verify_files(trial_dir, save_config, saved_scalars)
    if with_timestamp:
        check_tf_events(trial_dir, saved_scalars)


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
@pytest.mark.parametrize("with_timestamp", [True, False])
def test_tf_save_scalar(use_keras, collection, save_config, with_timestamp):
    helper_tensorflow_tests(use_keras, collection, save_config, with_timestamp)
    delete_local_trials([SMDEBUG_TF_HOOK_TESTS_DIR])
