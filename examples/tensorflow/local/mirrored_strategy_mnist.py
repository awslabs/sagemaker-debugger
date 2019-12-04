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

# Standard Library
import argparse

# Third Party
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

# First Party
import smdebug.tensorflow as smd

tf.logging.set_verbosity(tf.logging.INFO)


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
        dataset = dataset.batch(1)
        return dataset


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_cli_args():
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cmdline.add_argument(
        "--steps", type=int, default=20000, help="""Number of training steps to run."""
    )

    cmdline.add_argument("--save_all", type=str2bool, default=True)
    cmdline.add_argument("--out_dir", type=str)
    cmdline.add_argument("--save_frequency", type=int, help="How often to save TS data", default=10)
    cmdline.add_argument(
        "--reductions",
        type=str2bool,
        dest="reductions",
        default=False,
        help="save reductions of tensors instead of saving full tensors",
    )

    return cmdline


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == "GPU"])


def main(unused_argv):
    num_gpus = get_available_gpus()
    batch_size = 10 * num_gpus

    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()

    # input_fn which serves Dataset
    input_fn_provider = InputFnProvider(per_device_batch_size(batch_size, num_gpus))

    # Use multiple GPUs by MirroredStragtegy.
    # All avaiable GPUs will be used if `num_gpus` is omitted.
    if num_gpus > 1:
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
        print("### Doing Multi GPU Training")
    else:
        distribution = None
    # Pass to RunConfig
    config = tf.estimator.RunConfig(
        train_distribute=distribution, model_dir="/tmp/mnist_convnet_model"
    )

    # save tensors as reductions if necessary
    rdnc = (
        smd.ReductionConfig(reductions=["mean"], abs_reductions=["max"], norms=["l1"])
        if FLAGS.reductions
        else None
    )

    ts_hook = smd.SessionHook(
        out_dir=FLAGS.out_dir,
        save_all=FLAGS.save_all,
        include_collections=["weights", "gradients", "losses", "biases"],
        save_config=smd.SaveConfig(save_interval=FLAGS.save_frequency),
        reduction_config=rdnc,
    )

    ts_hook.set_mode(smd.modes.TRAIN)

    # Create the Estimator
    # pass RunConfig
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, config=config)

    # Train the model
    mnist_classifier.train(
        input_fn=input_fn_provider.train_input_fn, steps=FLAGS.steps, hooks=[ts_hook]
    )

    ts_hook.set_mode(smd.modes.EVAL)
    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        input_fn=input_fn_provider.eval_input_fn, hooks=[ts_hook]
    )
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
