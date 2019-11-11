# Standard Library
import argparse
import sys
import time
import uuid

# Third Party
import numpy as np
import tensorflow as tf

# First Party
from tornasole.tensorflow import SaveConfig, TornasoleHook, get_hook

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
parser.add_argument("--steps", type=int, help="Number of steps to run", default=100)
parser.add_argument("--scale", type=float, help="Scaling factor for inputs", default=1.0)
parser.add_argument(
    "--tornasole_frequency", type=float, help="How often to save TS data", default=10
)
parser.add_argument("--run_name", type=str, help="Run Name", default=str(uuid.uuid4()))
parser.add_argument("--local_reductions", nargs="+", type=str, default=[])
# running in Tf estimator mode, script need to accept --model_dir parameter
parser.add_argument("--model_dir", type=str, help="model dir", default=str(uuid.uuid4()))
args = parser.parse_args()

t = str(time.time())
hook = TornasoleHook(
    "s3://tornasolecodebuildtest/container_testing/ts_outputs/tf" + t,
    save_config=SaveConfig(save_interval=10),
)

# Network definition
with tf.name_scope("foobar"):
    x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    w = tf.Variable(initial_value=[[10.0], [10.0]])
with tf.name_scope("foobaz"):
    w0 = [[1], [1.0]]
    y = tf.matmul(x, w0)
loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")
global_step = tf.Variable(17, name="global_step", trainable=False)
increment_global_step_op = tf.assign(global_step, global_step + 1)
optimizer = tf.train.AdamOptimizer(args.lr)
optimizer = get_hook().wrap_optimizer(optimizer)
optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)
graph = tf.get_default_graph()
list_of_tuples = [op.outputs for op in graph.get_operations()]
sess = tf.train.MonitoredSession(hooks=[hook])
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
    print(f"Step={i}, Loss={_loss}")


# from tornasole.trials import create_trial
# tr = create_trial('s3://tornasolecodebuildtest/container_testing/ts_outputs/tf'+t)
# from tornasole.rules.generic import VanishingGradient
# r = VanishingGradient(tr)
# from tornasole.rules.rule_invoker import invoke_rule
# invoke_rule(r, start_step=0, end_step=80)
