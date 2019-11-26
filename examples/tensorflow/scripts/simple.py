# Standard Library
import argparse
import random

# Third Party
import numpy as np
import tensorflow as tf

# First Party
import smdebug.tensorflow as smd


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
parser.add_argument("--script-mode", type=str2bool, default=False)
parser.add_argument("--model_dir", type=str, help="S3 path for the model")
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
parser.add_argument("--steps", type=int, help="Number of steps to run", default=100)
parser.add_argument("--scale", type=float, help="Scaling factor for inputs", default=1.0)
parser.add_argument("--save_all", type=str2bool, default=True)
parser.add_argument("--smdebug_path", type=str, default="/opt/ml/output/tensors")
parser.add_argument("--save_frequency", type=int, help="How often to save TS data", default=10)
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
# for now, 2,2,12 could promise no assert failure when running tests
# if you wish to change the number, notice that certain steps' tensor value may be capable of variation
if args.random_seed:
    tf.set_random_seed(2)
    np.random.seed(2)
    random.seed(12)


if args.script_mode:
    # save tensors as reductions if necessary
    rdnc = (
        smd.ReductionConfig(reductions=["mean"], abs_reductions=["max"], norms=["l1"])
        if args.reductions
        else None
    )

    # create the hook
    # Note that we are saving all tensors here by passing save_all=True
    hook = smd.SessionHook(
        out_dir=args.smdebug_path,
        save_all=args.save_all,
        include_collections=["weights", "gradients", "losses"],
        save_config=smd.SaveConfig(save_interval=args.save_frequency),
        reduction_config=rdnc,
    )
    hooks = [hook]
else:
    hooks = []

# Network definition
# Note the use of name scopes
with tf.name_scope("foobar"):
    x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    w = tf.Variable(initial_value=[[10.0], [10.0]], name="weight1")
with tf.name_scope("foobaz"):
    w0 = [[1], [1.0]]
    y = tf.matmul(x, w0)
loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")

smd.get_hook("session", create_if_not_exists=True).add_to_collection("losses", loss)

global_step = tf.Variable(17, name="global_step", trainable=False)
increment_global_step_op = tf.assign(global_step, global_step + 1)

optimizer = tf.train.AdamOptimizer(args.lr)

if args.script_mode:
    # Wrap the optimizer with wrap_optimizer so Tornasole can find gradients and optimizer_variables to save
    optimizer = hook.wrap_optimizer(optimizer)

# use this wrapped optimizer to minimize loss
optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

if args.script_mode:
    hook.set_mode(smd.modes.TRAIN)

# pass the hook to hooks parameter of monitored session
sess = tf.train.MonitoredSession(hooks=hooks)

# use this session for running the tensorflow model
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
    print(f"Step={i}, Loss={_loss}")

if args.script_mode:
    hook.set_mode(smd.modes.EVAL)
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    sess.run([loss, increment_global_step_op], {x: x_})
