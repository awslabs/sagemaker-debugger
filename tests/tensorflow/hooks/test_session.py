# Third Party
import numpy as np
import tensorflow as tf

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
        smd.get_collection("losses").add(loss)
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
        assert len(tr.tensors())
