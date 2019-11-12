# Standard Library
import os
import shutil

# Third Party
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

# First Party
import smdebug.tensorflow as smd
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.locations import TensorFileLocation
from smdebug.core.reader import FileReader
from smdebug.tensorflow import reset_collections
from smdebug.tensorflow.session import SessionHook

# Local
from .utils import *


def helper_hook_write(data_dir, hook):
    learning_rate = 1e-3
    batch_size = 5

    # input
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    # ground truth output
    y = tf.placeholder(tf.float32, [None, 10], name="y")
    # weights
    W1 = tf.Variable(tf.random.normal([784, 300], stddev=1.0), name="w1")
    b1 = tf.Variable(tf.random.normal([300], stddev=1.0), name="b1")
    W2 = tf.Variable(tf.random.normal([300, 10], stddev=1.0), name="w2")
    b2 = tf.Variable(tf.random.normal([10], stddev=1.0), name="b2")

    # hidden layer, output activation functions
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)
    y_pred = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # clip values to ensure that cross-entropy loss can be evaluated
    y_clipped = tf.clip_by_value(y_pred, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(
        tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1), name="loss"
    )
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
        cross_entropy
    )

    init_op = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = 0.0

    # remove existing directory so that you can rerun and evaluate against a fresh run
    # instead of old weights saved to disk

    shutil.rmtree(data_dir, ignore_errors=True)

    # the variables we want to save
    saving_var_dict = {"w1": W1, "b1": b1, "w2": W2, "b2": b2}

    # train the network for 1000 iterations
    with tf.train.MonitoredSession(hooks=[hook]) as sess:
        for i in range(1000):
            x_in = np.random.normal(0, 1, (batch_size, 784))
            y_truth = np.random.randint(0, 2, (batch_size, 10))
            feed = {x: x_in, y: y_truth}
            tensor_dict = {
                "opt": optimizer,
                "w1": W1,
                "b1": b1,
                "w2": W2,
                "b2": b2,
                "loss": cross_entropy,
            }
            v = sess.run(tensor_dict, feed_dict=feed)

            # track, output loss. currently has random initialization, so it's not that useful

            loss += v["loss"] / 1000

    # read saved weights from disk using summary iterator, verify if in-memory weights at end of training
    # are identical to the ones we have saved using SessionHook
    step_dir = TensorFileLocation.get_step_dir_path(data_dir, 999)
    files = os.listdir(step_dir)
    print(v.keys())
    for f in files:
        fr = FileReader(os.path.join(step_dir, f))
        for tupv in fr.read_tensors():
            (tensor_name, step, tensor_data, mode, mode_step) = tupv
            if tensor_name in v:
                assert np.allclose(tensor_data, v[tensor_name])


@pytest.mark.slow  # 0:01 to run
def test_hook_write(out_dir):
    pre_test_clean_up()
    # set up hook
    hook = SessionHook(
        out_dir, save_all=True, include_collections=None, save_config=SaveConfig(save_interval=999)
    )
    helper_hook_write(out_dir, hook)


def test_hook_write_json(out_dir, monkeypatch):
    pre_test_clean_up()
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR, "tests/tensorflow/hooks/test_json_configs/test_write.json"
    )
    hook = smd.SessionHook.hook_from_config()
    helper_hook_write(out_dir, hook)
