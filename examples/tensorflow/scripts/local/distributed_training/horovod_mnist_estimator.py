#  Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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
import errno
import os

# Third Party
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from tensorflow import keras

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Horovod: scale learning rate by the number of workers.
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001 * hvd.size(), momentum=0.9)

        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)

        # Add smdebug Optimizer
        optimizer = smd.get_hook().wrap_optimizer(optimizer)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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
    cmdline.add_argument("--smdebug_path", type=str, default="/opt/ml/output/tensors")
    cmdline.add_argument("--save_frequency", type=int, help="How often to save TS data", default=10)
    cmdline.add_argument(
        "--reductions",
        type=str2bool,
        dest="reductions",
        default=False,
        help="save reductions of tensors instead of saving full tensors",
    )

    return cmdline


def main(unused_argv):
    # Get commandline args

    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()

    # Horovod: initialize Horovod.
    hvd.init()

    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    # Download and load MNIST dataset.
    (train_data, train_labels), (eval_data, eval_labels) = keras.datasets.mnist.load_data(
        "/tmp/MNIST-data-%d" % hvd.rank()
    )

    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    train_data = np.reshape(train_data, (-1, 784)) / 255.0
    eval_data = np.reshape(eval_data, (-1, 784)) / 255.0

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = "./mnist_convnet_model" if hvd.rank() == 0 else None

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(session_config=config),
    )

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True
    )

    # Setup the Tornasole Hook

    # save tensors as reductions if necessary
    rdnc = (
        smd.ReductionConfig(reductions=["mean"], abs_reductions=["max"], norms=["l1"])
        if FLAGS.reductions
        else None
    )

    ts_hook = smd.SessionHook(
        out_dir=FLAGS.smdebug_path,
        save_all=FLAGS.save_all,
        include_collections=["weights", "gradients", "losses", "biases"],
        save_config=smd.SaveConfig(save_interval=FLAGS.save_frequency),
        reduction_config=rdnc,
    )

    ts_hook.set_mode(smd.modes.TRAIN)

    # Horovod: adjust number of steps based on number of GPUs.
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=FLAGS.steps // hvd.size(),
        hooks=[logging_hook, bcast_hook, ts_hook],
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False
    )

    ts_hook.set_mode(smd.modes.EVAL)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn, hooks=[ts_hook])
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
