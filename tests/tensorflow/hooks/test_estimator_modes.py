"""
WARNING: This test file used to take 25 minutes.
I've made some changes to speed it up now (~3 minutes),
but choose fast defaults moving forward so CI runs quickly.

All other TF tests combined run in <30 seconds, so would be
nice if we could speed up the S3 integration testing.

Integration tests with S3 take 95% of the time.
"""

# Standard Library
import shutil
from datetime import datetime

# Third Party
import numpy as np
import pytest
import tensorflow as tf
from tests.analysis.utils import delete_s3_prefix
from tests.utils import verify_shapes

# First Party
import smdebug.tensorflow as smd
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.utils import is_s3
from smdebug.tensorflow.session import SessionHook
from smdebug.trials import create_trial


def help_test_mnist(
    path,
    save_config=None,
    reduction_config=None,
    hook=None,
    set_modes=True,
    num_steps=10,
    num_eval_steps=None,
    save_all=False,
    steps=None,
    include_collections=None,
):
    trial_dir = path
    tf.reset_default_graph()

    def cnn_model_fn(features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
        )

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            optimizer = smd.get_hook().wrap_optimizer(optimizer)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Load training and eval data
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)  # not required

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model"
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=2, num_epochs=None, shuffle=True
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, batch_size=1, shuffle=False
    )

    if hook is None:
        if include_collections is None:
            include_collections = ["weights", "gradients", "default", "losses"]
        hook = smd.SessionHook(
            out_dir=trial_dir,
            save_config=save_config,
            include_collections=include_collections,
            save_all=save_all,
            reduction_config=reduction_config,
        )

    if num_eval_steps is None:
        num_eval_steps = num_steps

    def train(num_steps):
        if set_modes:
            hook.set_mode(smd.modes.TRAIN)
        mnist_classifier.train(input_fn=train_input_fn, steps=num_steps, hooks=[hook])

    def evaluate(num_eval_steps):
        if set_modes:
            hook.set_mode(smd.modes.EVAL)
        mnist_classifier.evaluate(input_fn=eval_input_fn, steps=num_eval_steps, hooks=[hook])

    # def train_and_evaluate(num_steps, num_eval_steps):
    #     tf.estimator.train_and_evaluate(
    #         mnist_classifier,
    #         train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_steps),#, hooks=[hook]),
    #         eval_spec=tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=num_eval_steps)#, hooks=[hook]),
    #     )

    if steps is None:
        steps = ["train", "eval", "train"]

    for s in steps:
        if s == "train":
            # train one step and display the probabilties
            train(num_steps)
        elif s == "eval":
            evaluate(num_eval_steps)
        # elif s == "traineval":
        #     train_and_evaluate(num_steps, num_eval_steps)

    hook.close()


def helper_test_mnist_trial(trial_dir):
    tr = create_trial(trial_dir)
    assert len(tr.steps()) == 3
    assert len(tr.steps(mode=smd.modes.TRAIN)) == 2
    assert len(tr.steps(mode=smd.modes.EVAL)) == 1
    assert len(tr.tensor_names()) == 13
    on_s3, bucket, prefix = is_s3(trial_dir)
    if not on_s3:
        shutil.rmtree(trial_dir, ignore_errors=True)
    else:
        delete_s3_prefix(bucket, prefix)


@pytest.mark.slow  # 0:02 to run
def test_mnist(out_dir, on_s3=False):
    if on_s3:
        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        bucket = "smdebug-testing"
        prefix = "outputs/hooks/estimator_modes/" + run_id
        out_dir = f"s3://{bucket}/{prefix}"
    help_test_mnist(out_dir, save_config=smd.SaveConfig(save_interval=2), num_steps=2, steps=None)
    helper_test_mnist_trial(out_dir)


@pytest.mark.slow  # 0:02 to run
def test_mnist_shapes(out_dir, on_s3=False):
    if on_s3:
        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        bucket = "smdebug-testing"
        prefix = "outputs/hooks/estimator_modes/" + run_id
        out_dir = f"s3://{bucket}/{prefix}"
    help_test_mnist(
        out_dir,
        save_all=True,
        save_config=smd.SaveConfig(save_steps=[0]),
        num_steps=1,
        steps=None,
        reduction_config=smd.ReductionConfig(save_shape=True),
    )
    verify_shapes(out_dir, 0)


@pytest.mark.slow  # 0:02 to run
def test_mnist_shapes_s3(out_dir):
    test_mnist_shapes(out_dir, on_s3=True)


@pytest.mark.slow  # 0:02 to run
def test_mnist_local_json(out_dir, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR, "tests/tensorflow/hooks/test_json_configs/test_mnist_local.json"
    )
    hook = SessionHook.create_from_json_file()
    help_test_mnist(path=out_dir, hook=hook, num_steps=2)
    helper_test_mnist_trial(out_dir)


@pytest.mark.slow  # 1:04 to run
def test_mnist_s3(out_dir):
    # Takes 1:04 to run, compared to 4 seconds above.
    # Speed improvements, or should we migrate integration tests to their own folder?
    test_mnist(out_dir, True)


def helper_test_multi_save_configs_trial(trial_dir):
    tr = create_trial(trial_dir)
    print(tr.steps(), tr.steps(mode=smd.modes.TRAIN), tr.steps(mode=smd.modes.EVAL))
    assert len(tr.steps()) == 4
    assert len(tr.steps(mode=smd.modes.TRAIN)) == 3
    assert len(tr.steps(mode=smd.modes.EVAL)) == 1
    assert len(tr.tensor_names()) == 1
    on_s3, bucket, prefix = is_s3(trial_dir)
    if not on_s3:
        shutil.rmtree(trial_dir)
    else:
        delete_s3_prefix(bucket, prefix)


@pytest.mark.slow  # 0:04 to run
def test_mnist_local_multi_save_configs(out_dir, on_s3=False):
    # Runs in 0:04
    if on_s3:
        run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
        bucket = "smdebug-testing"
        prefix = "outputs/hooks/estimator_modes/" + run_id
        out_dir = f"s3://{bucket}/{prefix}"
    help_test_mnist(
        out_dir,
        smd.SaveConfig(
            {
                smd.modes.TRAIN: smd.SaveConfigMode(save_interval=2),
                smd.modes.EVAL: smd.SaveConfigMode(save_interval=3),
            }
        ),
        include_collections=["losses"],
        num_steps=3,
    )
    helper_test_multi_save_configs_trial(out_dir)


@pytest.mark.slow  # 0:52 to run
def test_mnist_s3_multi_save_configs(out_dir):
    # Takes 0:52 to run, compared to 4 seconds above. Speed improvements?
    test_mnist_local_multi_save_configs(out_dir, True)


@pytest.mark.slow  # 0:02 to run
def test_mnist_local_multi_save_configs_json(out_dir, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_save_config_modes_hook_config.json",
    )
    hook = smd.SessionHook.create_from_json_file()
    help_test_mnist(out_dir, hook=hook, num_steps=3)
    helper_test_multi_save_configs_trial(out_dir)


def test_mode_changes(out_dir):
    help_test_mnist(
        out_dir,
        save_config=smd.SaveConfig(save_interval=2),
        num_steps=2,
        steps=["train", "eval", "train", "eval", "train", "train"],
    )
    tr = create_trial(out_dir)
    print(tr.steps(), tr.steps(mode=smd.modes.TRAIN), tr.steps(mode=smd.modes.EVAL))
    assert len(tr.steps()) == 6
    assert len(tr.steps(mode=smd.modes.TRAIN)) == 4
    assert len(tr.steps(mode=smd.modes.EVAL)) == 2
    assert len(tr.tensor_names()) == 13
