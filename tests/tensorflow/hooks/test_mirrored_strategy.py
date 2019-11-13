#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

# Future
from __future__ import absolute_import, division, print_function

# Third Party
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.client import device_lib
from tests.tensorflow.utils import create_trial_fast_refresh

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.core.modes import ModeKeys
from smdebug.exceptions import TensorUnavailableForStep


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
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


def per_device_batch_size(batch_size, num_gpus):
    """For multi-gpu, batch-size must be a multiple of the number of GPUs.
    Note that this should eventually be handled by DistributionStrategies
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    Args:
      batch_size: Global batch size to be divided among devices. This should be
        equal to num_gpus times the single-GPU batch_size for multi-gpu training.
      num_gpus: How many GPUs are used with DistributionStrategies.
    Returns:
      Batch size per device.
    Raises:
      ValueError: if batch_size is not divisible by number of devices
    """
    if num_gpus <= 1:
        return batch_size

    remainder = batch_size % num_gpus
    if remainder:
        err = (
            "When running with multiple GPUs, batch size "
            "must be a multiple of the number of available GPUs. Found {} "
            "GPUs with a batch size of {}; try --batch_size={} instead."
        ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)
    return int(batch_size / num_gpus)


class InputFnProvider:
    def __init__(self, train_batch_size):
        self.train_batch_size = train_batch_size
        self.__load_data()

    def __load_data(self):
        # Load training and eval data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images  # Returns np.array
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.eval_data = mnist.test.images  # Returns np.array
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    def train_input_fn(self):
        """An input function for training"""
        # Shuffle, repeat, and batch the examples.
        dataset = tf.data.Dataset.from_tensor_slices(({"x": self.train_data}, self.train_labels))
        dataset = dataset.shuffle(1000).repeat().batch(self.train_batch_size)
        return dataset

    def eval_input_fn(self):
        """An input function for evaluation or prediction"""
        dataset = tf.data.Dataset.from_tensor_slices(({"x": self.eval_data}, self.eval_labels))
        dataset = dataset.batch(1).repeat()
        return dataset


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == "GPU"])


def helper_mirrored(
    trial_dir,
    save_all=False,
    num_steps=3,
    save_config=None,
    reduction_config=None,
    include_collections=None,
    steps=None,
    eval_distributed=False,
):
    num_gpus = get_available_gpus()
    num_devices = num_gpus if num_gpus > 0 else 1
    batch_size = 10 * num_devices

    smd.reset_collections()

    # input_fn which serves Dataset
    input_fn_provider = InputFnProvider(per_device_batch_size(batch_size, num_devices))

    # Use multiple GPUs by MirroredStragtegy.
    # All avaiable GPUs will be used if `num_gpus` is omitted.
    # if num_devices > 1:
    distribution = tf.contrib.distribute.MirroredStrategy()
    # print("### Doing Multi GPU Training")
    # else:
    #     distribution = None
    # Pass to RunConfig
    config = tf.estimator.RunConfig(
        train_distribute=distribution,
        eval_distribute=distribution if eval_distributed else None,
        model_dir="/tmp/mnist_convnet_model",
    )

    if save_config is None:
        save_config = smd.SaveConfig(save_interval=2)

    if include_collections is None:
        include_collections = [
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.GRADIENTS,
            CollectionKeys.LOSSES,
        ]

    ts_hook = smd.SessionHook(
        out_dir=trial_dir,
        save_all=save_all,
        include_collections=include_collections,
        save_config=save_config,
        reduction_config=reduction_config,
    )

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, config=config)
    if steps is None:
        steps = ["train"]

    for s in steps:
        if s == "train":
            print("Starting train")
            ts_hook.set_mode(smd.modes.TRAIN)
            # Train the model
            mnist_classifier.train(
                input_fn=input_fn_provider.train_input_fn, steps=num_steps, hooks=[ts_hook]
            )
        elif s == "eval":
            print("Starting eval")
            ts_hook.set_mode(smd.modes.EVAL)
            # Evaluate the model and print results
            mnist_classifier.evaluate(
                input_fn=input_fn_provider.eval_input_fn, steps=num_steps, hooks=[ts_hook]
            )
        elif s == "predict":
            ts_hook.set_mode(smd.modes.PREDICT)
            # Evaluate the model and print results
            print("Starting predict")
            p = mnist_classifier.predict(input_fn=input_fn_provider.eval_input_fn, hooks=[ts_hook])
            for i in range(num_steps):
                print(next(p))
    ts_hook._cleanup()
    return distribution


@pytest.mark.slow
def test_basic(out_dir):
    strategy = helper_mirrored(
        out_dir,
        steps=["train", "eval", "predict", "train"],
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.GRADIENTS,
            CollectionKeys.LOSSES,
        ],
        eval_distributed=False,
    )
    tr = create_trial_fast_refresh(out_dir)
    # wts, grads, losses
    print(tr.tensors())
    assert len(tr.tensors()) == 8 + 8 + (1 * strategy.num_replicas_in_sync) + 1
    assert len(tr.steps()) == 7
    assert len(tr.steps(ModeKeys.TRAIN)) == 3
    assert len(tr.steps(ModeKeys.EVAL)) == 2
    assert len(tr.steps(ModeKeys.PREDICT)) == 2

    for tname in tr.tensors(collection="weights"):
        for s in tr.tensor(tname).steps(ModeKeys.TRAIN):
            assert len(tr.tensor(tname).workers(s, ModeKeys.TRAIN)) == strategy.num_replicas_in_sync
            for worker in tr.tensor(tname).workers(s, ModeKeys.TRAIN):
                assert tr.tensor(tname).value(s, worker=worker, mode=ModeKeys.TRAIN) is not None
        for s in tr.tensor(tname).steps(ModeKeys.EVAL):
            assert len(tr.tensor(tname).workers(s, ModeKeys.EVAL)) == strategy.num_replicas_in_sync
            assert tr.tensor(tname).value(s, mode=ModeKeys.EVAL) is not None

    for s in tr.tensor("Identity_2:0").steps(ModeKeys.TRAIN):
        for w in tr.tensor("Identity_2:0").workers(s, ModeKeys.TRAIN):
            assert tr.tensor("Identity_2:0").value(s, worker=w, mode=ModeKeys.TRAIN) is not None
        assert (
            len(tr.tensor("Identity_2:0").workers(s, ModeKeys.TRAIN))
            == strategy.num_replicas_in_sync
        )

    for tname in tr.tensors(collection="losses"):
        if tname != "Identity_2:0":
            for s in tr.tensor(tname).steps(ModeKeys.TRAIN):
                assert len(tr.tensor(tname).workers(s, ModeKeys.TRAIN)) == 1
                assert tr.tensor(tname).value(s, mode=ModeKeys.TRAIN) is not None

    tname = "sparse_softmax_cross_entropy_loss/value:0"
    for s in tr.tensor(tname).steps(ModeKeys.EVAL):
        assert len(tr.tensor(tname).workers(s, ModeKeys.EVAL)) == strategy.num_replicas_in_sync
        assert tr.tensor(tname).value(s, mode=ModeKeys.EVAL) is not None


@pytest.mark.slow
def test_eval_distributed(out_dir):
    strategy = helper_mirrored(
        out_dir,
        steps=["train", "eval"],
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.BIASES, CollectionKeys.LOSSES],
        eval_distributed=True,
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensors()) == 8 + 1 * strategy.num_replicas_in_sync + 1
    assert len(tr.steps()) == 4
    assert len(tr.steps(ModeKeys.TRAIN)) == 2
    assert len(tr.steps(ModeKeys.EVAL)) == 2

    for tname in tr.tensors(collection="weights"):
        for s in tr.tensor(tname).steps(ModeKeys.TRAIN):
            assert len(tr.tensor(tname).workers(s, ModeKeys.TRAIN)) == strategy.num_replicas_in_sync
            for worker in tr.tensor(tname).workers(s, ModeKeys.TRAIN):
                assert tr.tensor(tname).value(s, worker=worker, mode=ModeKeys.TRAIN) is not None
        for s in tr.tensor(tname).steps(ModeKeys.EVAL):
            assert len(tr.tensor(tname).workers(s, ModeKeys.EVAL)) == strategy.num_replicas_in_sync
            assert tr.tensor(tname).value(s, mode=ModeKeys.EVAL) is not None

    for s in tr.tensor("Identity_2:0").steps(ModeKeys.TRAIN):
        for w in tr.tensor("Identity_2:0").workers(s, ModeKeys.TRAIN):
            assert tr.tensor("Identity_2:0").value(s, worker=w, mode=ModeKeys.TRAIN) is not None
        assert (
            len(tr.tensor("Identity_2:0").workers(s, ModeKeys.TRAIN))
            == strategy.num_replicas_in_sync
        )

    for tname in tr.tensors(collection="losses"):
        for s in tr.tensor(tname).steps(ModeKeys.EVAL):
            assert len(tr.tensor(tname).workers(s, ModeKeys.EVAL)) == 1
            assert tr.tensor(tname).value(s, mode=ModeKeys.EVAL) is not None
        if tname != "Identity_2:0":
            for s in tr.tensor(tname).steps(ModeKeys.TRAIN):
                assert len(tr.tensor(tname).workers(s, ModeKeys.EVAL)) == 1
                assert tr.tensor(tname).value(s, mode=ModeKeys.EVAL) is not None


@pytest.mark.slow
def test_reductions(out_dir):
    strategy = helper_mirrored(
        out_dir,
        steps=["train", "eval"],
        reduction_config=smd.ReductionConfig(
            reductions=["sum", "max"], abs_reductions=["sum", "max"], norms=["l1"]
        ),
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.BIASES, CollectionKeys.LOSSES],
        eval_distributed=True,
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensors()) == 8 + 1 * strategy.num_replicas_in_sync + 1
    assert len(tr.steps()) == 4
    assert len(tr.steps(ModeKeys.TRAIN)) == 2
    assert len(tr.steps(ModeKeys.EVAL)) == 2

    for tname in tr.tensors(collection="weights"):
        for s in tr.tensor(tname).steps(ModeKeys.TRAIN):
            try:
                tr.tensor(tname).value(s, mode=ModeKeys.TRAIN)
                assert False
            except TensorUnavailableForStep:
                assert len(tr.tensor(tname).reduction_values(s, mode=ModeKeys.TRAIN)) == 5

        for s in tr.tensor(tname).steps(ModeKeys.EVAL):
            try:
                tr.tensor(tname).value(s, mode=ModeKeys.EVAL)
                assert False
            except TensorUnavailableForStep:
                assert len(tr.tensor(tname).reduction_values(s, mode=ModeKeys.EVAL)) == 5

    for tname in tr.tensors(collection="losses"):
        for s in tr.tensor(tname).steps(ModeKeys.EVAL):
            assert len(tr.tensor(tname).reduction_values(s, mode=ModeKeys.EVAL)) == 0
            assert tr.tensor(tname).value(s, mode=ModeKeys.EVAL) is not None

    for tname in tr.tensors(collection="losses"):
        for s in tr.tensor(tname).steps(ModeKeys.TRAIN):
            assert len(tr.tensor(tname).reduction_values(s, mode=ModeKeys.TRAIN)) == 0
            assert tr.tensor(tname).value(s, mode=ModeKeys.TRAIN) is not None


@pytest.mark.slow
def test_save_all(out_dir):
    strategy = helper_mirrored(
        out_dir, steps=["train"], num_steps=1, save_all=True, eval_distributed=True
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensors()) > 100
