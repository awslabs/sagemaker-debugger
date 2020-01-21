# Standard Library
import os
from os.path import isfile, join

# Third Party
import numpy as np
import tensorflow.compat.v1 as tf

# First Party
from smdebug.core.utils import get_path_to_collections


def simple_model(hook, steps=10, lr=0.4):
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

    for i in range(steps):
        x_ = np.random.random((10, 2)) * 0.1
        _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
        print(f"Step={i}, Loss={_loss}")

    sess.close()


def get_dirs_files(path):
    entries = os.listdir(path)
    onlyfiles = [f for f in entries if isfile(join(path, f))]
    subdirs = [x for x in entries if x not in onlyfiles]
    return subdirs, onlyfiles


def get_collection_files(path):
    path = get_path_to_collections(path)
    entries = os.listdir(path)
    files = [f for f in entries if isfile(join(path, f))]
    return files


def get_event_file_path_length(dir_path):
    files = os.listdir(dir_path)
    if len(files) > 0:
        filepath = join(dir_path, files[0])
        return filepath, os.stat(filepath).st_size
    else:
        return None, 0


def pre_test_clean_up():
    tf.reset_default_graph()
    tf.set_random_seed(1)
    np.random.seed(1)
