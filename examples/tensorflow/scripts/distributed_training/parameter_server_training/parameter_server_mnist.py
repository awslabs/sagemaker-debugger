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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tornasole.tensorflow as ts
import argparse
import json
import os

from tensorflow.python.client import device_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

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
        optimizer = ts.get_hook().wrap_optimizer(optimizer)
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
    cmdline.add_argument("--tornasole_path", type=str, default="/opt/ml/output/tensors")
    cmdline.add_argument(
        "--tornasole_frequency", type=int, help="How often to save TS data", default=10
    )
    cmdline.add_argument(
        "--reductions",
        type=str2bool,
        dest="reductions",
        default=False,
        help="save reductions of tensors instead of saving full tensors",
    )

    cmdline.add_argument(
        "--node_type", type=str, required=True, dest="node_type", help="node type: worker or ps"
    )

    cmdline.add_argument(
        "--task_index", type=int, required=True, dest="task_index", help="task index"
    )

    cmdline.add_argument(
        "--hostfile",
        default=None,
        type=str,
        required=False,
        dest="hostfile",
        help="Path to hostfile",
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

    # Use multiple GPUs by ParameterServerStrategy.
    # All avaiable GPUs will be used if `num_gpus` is omitted.

    if num_gpus > 1:
        strategy = tf.distribute.experimental.ParameterServerStrategy()
        if not os.getenv("TF_CONFIG"):
            if FLAGS.hostfile is None:
                raise Exception("--hostfile not provided and TF_CONFIG not set. Please do either.")
            nodes = list()
            try:
                f = open(FLAGS.hostfile)
                for line in f.readlines():
                    nodes.append(line.strip())
            except OSError as e:
                print(e.errno)

            os.environ["TF_CONFIG"] = json.dumps(
                {
                    "cluster": {"worker": [nodes[0], nodes[1]], "ps": [nodes[2]]},
                    "task": {"type": FLAGS.node_type, "index": FLAGS.task_index},
                }
            )

        print("### Doing Multi GPU Training")
    else:
        strategy = None
    # Pass to RunConfig
    config = tf.estimator.RunConfig(train_distribute=strategy)

    # save tensors as reductions if necessary
    rdnc = (
        ts.ReductionConfig(reductions=["mean"], abs_reductions=["max"], norms=["l1"])
        if FLAGS.reductions
        else None
    )

    ts_hook = ts.TornasoleHook(
        out_dir=FLAGS.tornasole_path,
        save_all=FLAGS.save_all,
        include_collections=["weights", "gradients", "losses", "biases"],
        save_config=ts.SaveConfig(save_interval=FLAGS.tornasole_frequency),
        reduction_config=rdnc,
    )

    ts_hook.set_mode(ts.modes.TRAIN)

    # Create the Estimator
    # pass RunConfig
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, config=config)

    hooks = list()
    hooks.append(ts_hook)

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn_provider.train_input_fn, max_steps=FLAGS.steps, hooks=hooks
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn_provider.eval_input_fn, steps=FLAGS.steps, hooks=hooks
    )

    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(input_fn=input_fn_provider.eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
