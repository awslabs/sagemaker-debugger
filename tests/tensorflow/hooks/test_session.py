# Third Party
import numpy as np
import tensorflow as tf
from tests.zero_code_change.tf_utils import get_data, get_train_op_and_placeholders

# First Party
import smdebug.tensorflow as smd
from smdebug.trials import create_trial


def test_new_graph(out_dir):
    # tests that we can correctly interpret an explicitly created graph
    g1 = tf.get_default_graph()
    g = tf.Graph()
    with g.as_default():
        assert g != g1
        assert g == tf.get_default_graph()
        hook = smd.SessionHook(
            out_dir,
            include_collections=["weights", "losses", "scalars"],
            save_config=smd.SaveConfig(save_steps=[0, 1, 2, 3]),
        )
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

        optimizer = tf.train.AdamOptimizer(0.1)
        optimizer = hook.wrap_optimizer(optimizer)
        optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)
        sess = tf.train.MonitoredSession(hooks=[hook])
        for i in range(5):
            x_ = np.random.random((10, 2)) * 0.1
            sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
        sess.close()
        tr = create_trial(out_dir)
        assert len(tr.tensor_names())


def test_uninit_sess_run(out_dir):
    train_op, X, Y = get_train_op_and_placeholders()
    init = tf.global_variables_initializer()
    mnist = get_data()
    hook = smd.SessionHook(out_dir, include_collections=["weights"])
    sess = tf.train.MonitoredSession(hooks=[hook])

    with sess:
        sess.run(init)
        for step in range(1, 101):
            batch_x, batch_y = mnist.train.next_batch(32)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

    # Check that hook created and tensors saved
    trial = smd.create_trial(path=out_dir)
    assert len(trial.steps()) > 0, "Nothing saved at any step."
    assert len(trial.tensor_names()) > 0, "Tensors were not saved."
    assert len(trial.tensor_names(collection="weights")) > 0
