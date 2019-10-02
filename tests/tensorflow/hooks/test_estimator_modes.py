"""
WARNING: This test file used to take 25 minutes.
I've made some changes to speed it up now (~3 minutes),
but choose fast defaults moving forward so CI runs quickly.

All other TF tests combined run in <30 seconds, so would be
nice if we could speed up the S3 integration testing.

Integration tests with S3 take 95% of the time.
"""


import pytest
import tensorflow as tf
import numpy as np
import shutil
import os
from datetime import datetime
from .utils import TORNASOLE_TF_HOOK_TESTS_DIR
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR

import tornasole.tensorflow as ts
from tornasole.tensorflow import reset_collections
from tornasole.tensorflow.hook import TornasoleHook
from tornasole.trials import create_trial
from tests.analysis.utils import delete_s3_prefix

def help_test_mnist(path, save_config=None, hook=None, set_modes=True):
    trial_dir = path
    tf.reset_default_graph()
    if hook is None:
        reset_collections()

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
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            optimizer = ts.TornasoleOptimizer(optimizer)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def train(num_steps):
        mnist_classifier.train(
                input_fn=train_input_fn,
                steps=num_steps,
                hooks=[hook])

    # Load training and eval data
    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)  # not required

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=2,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        batch_size=1,
        shuffle=False)
    if hook is None:
        hook = ts.TornasoleHook(out_dir=trial_dir,
                                save_config=save_config)

    if set_modes:
        hook.set_mode(ts.modes.TRAIN)
    # train one step and display the probabilties
    train(2)

    if set_modes:
        hook.set_mode(ts.modes.EVAL)
    mnist_classifier.evaluate(input_fn=eval_input_fn,
                              steps=3,
                              hooks=[hook])

    if set_modes:
        hook.set_mode(ts.modes.TRAIN)
    train(2)

    return train

@pytest.mark.slow # 0:02 to run
def test_mnist_local():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    help_test_mnist(trial_dir, ts.SaveConfig(save_interval=2))
    tr = create_trial(trial_dir)
    assert len(tr.available_steps()) == 4
    assert len(tr.available_steps(mode=ts.modes.TRAIN)) == 2
    assert len(tr.available_steps(mode=ts.modes.EVAL)) == 2
    assert len(tr.tensors()) == 17
    shutil.rmtree(trial_dir)

@pytest.mark.slow # 0:02 to run
def test_mnist_local_json():
    out_dir = 'newlogsRunTest1/test_mnist_local_json_config'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/tensorflow/hooks/test_json_configs/test_mnist_local.json'
    hook = TornasoleHook.hook_from_config()
    help_test_mnist(path=out_dir, hook=hook)
    tr = create_trial(out_dir)
    assert len(tr.available_steps()) == 4
    assert len(tr.available_steps(mode=ts.modes.TRAIN)) == 2
    assert len(tr.available_steps(mode=ts.modes.EVAL)) == 2
    assert len(tr.tensors()) == 17
    shutil.rmtree(out_dir, ignore_errors=True)

@pytest.mark.slow # 1:04 to run
def test_mnist_s3():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    bucket = 'tornasole-testing'
    prefix = 'tornasole_tf/hooks/estimator_modes/' + run_id
    trial_dir = f's3://{bucket}/{prefix}'
    help_test_mnist(trial_dir, ts.SaveConfig(save_interval=2))
    tr = create_trial(trial_dir)
    assert len(tr.available_steps()) == 4
    assert len(tr.available_steps(mode=ts.modes.TRAIN)) == 2
    assert len(tr.available_steps(mode=ts.modes.EVAL)) == 2
    assert len(tr.tensors()) == 17
    delete_s3_prefix(bucket, prefix)

@pytest.mark.slow # 0:04 to run
def test_mnist_local_multi_save_configs():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    help_test_mnist(trial_dir, ts.SaveConfig({
        ts.modes.TRAIN: ts.SaveConfigMode(save_interval=2),
        ts.modes.EVAL: ts.SaveConfigMode(save_interval=3)
    }))
    tr = create_trial(trial_dir)
    assert len(tr.available_steps()) == 3
    assert len(tr.available_steps(mode=ts.modes.TRAIN)) == 2
    assert len(tr.available_steps(mode=ts.modes.EVAL)) == 1
    assert len(tr.tensors()) == 17
    shutil.rmtree(trial_dir)

@pytest.mark.slow # 0:52 to run
def test_mnist_s3_multi_save_configs():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    bucket = 'tornasole-testing'
    prefix = 'tornasole_tf/hooks/estimator_modes/' + run_id
    trial_dir = f's3://{bucket}/{prefix}'
    help_test_mnist(trial_dir, ts.SaveConfig({
        ts.modes.TRAIN: ts.SaveConfigMode(save_interval=2),
        ts.modes.EVAL: ts.SaveConfigMode(save_interval=3)
    }))
    tr = create_trial(trial_dir)
    assert len(tr.available_steps()) == 3
    assert len(tr.available_steps(mode=ts.modes.TRAIN)) == 2
    assert len(tr.available_steps(mode=ts.modes.EVAL)) == 1
    assert len(tr.tensors()) == 17
    delete_s3_prefix(bucket, prefix)

@pytest.mark.slow # 0:02 to run
def test_mnist_local_multi_save_configs_json():
    out_dir = 'newlogsRunTest1/test_save_config_modes_hook_config'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/tensorflow/hooks/test_json_configs/test_save_config_modes_hook_config.json'
    hook = ts.TornasoleHook.hook_from_config()
    help_test_mnist(out_dir, hook=hook)
    tr = create_trial(out_dir)
    assert len(tr.available_steps()) == 3
    assert len(tr.available_steps(mode=ts.modes.TRAIN)) == 2
    assert len(tr.available_steps(mode=ts.modes.EVAL)) == 1
    assert len(tr.tensors()) == 17
    shutil.rmtree(out_dir)
