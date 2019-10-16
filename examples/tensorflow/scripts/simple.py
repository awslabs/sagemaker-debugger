import argparse
import numpy as np
import tensorflow as tf
import tornasole.tensorflow as ts
import random


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="S3 path for the model")
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
parser.add_argument("--steps", type=int, help="Number of steps to run", default=100)
parser.add_argument("--scale", type=float, help="Scaling factor for inputs", default=1.0)
parser.add_argument("--save_all", type=str2bool, default=True)
parser.add_argument("--tornasole_path", type=str, default="/opt/ml/output/tensors")
parser.add_argument("--tornasole_frequency", type=int, help="How often to save TS data", default=10)
parser.add_argument("--random_seed", type=bool, default=False)
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument(
    "--reductions",
    dest="reductions",
    action="store_true",
    help="save reductions of tensors instead of saving full tensors",
)
feature_parser.add_argument(
    "--no_reductions", dest="reductions", action="store_false", help="save full tensors"
)
args = parser.parse_args()

# these random seeds are only intended for test purpose.
# for now, 2,2,12 could promise no assert failure when running tornasole_rules test_rules.py with config.yaml
# if you wish to change the number, notice that certain steps' tensor value may be capable of variation
if args.random_seed:
    tf.set_random_seed(2)
    np.random.seed(2)
    random.seed(12)

# Network definition
# Note the use of name scopes
with tf.name_scope("foobar"):
    x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    w = tf.Variable(initial_value=[[10.0], [10.0]], name="weight1")
    tf.summary.histogram("weight1_summ", w)
with tf.name_scope("foobaz"):
    w0 = [[1], [1.0]]
    y = tf.matmul(x, w0)
loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")
ts.add_to_collection("losses", loss)
tf.summary.scalar("loss_summ", loss)

global_step = tf.Variable(17, name="global_step", trainable=False)
increment_global_step_op = tf.assign(global_step, global_step + 1)

summ = tf.summary.merge_all()

optimizer = tf.train.AdamOptimizer(args.lr)

# Wrap the optimizer with TornasoleOptimizer so Tornasole can find gradients and optimizer_variables to save
optimizer = ts.TornasoleOptimizer(optimizer)

# use this wrapped optimizer to minimize loss
optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

# save tensors as reductions if necessary
rdnc = (
    ts.ReductionConfig(reductions=["mean"], abs_reductions=["max"], norms=["l1"])
    if args.reductions
    else None
)

# create the hook
# Note that we are saving all tensors here by passing save_all=True
hook = ts.TornasoleHook(
    out_dir=args.tornasole_path,
    save_all=args.save_all,
    include_collections=["weights", "gradients", "losses"],
    save_config=ts.SaveConfig(save_interval=args.tornasole_frequency),
    reduction_config=rdnc,
)

hook.set_mode(ts.modes.TRAIN)

# pass the hook to hooks parameter of monitored session
sess = tf.train.MonitoredSession(hooks=[hook])

# use this session for running the tensorflow model
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
    print(f"Step={i}, Loss={_loss}")

hook.set_mode(ts.modes.EVAL)
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    sess.run([loss, increment_global_step_op], {x: x_})
