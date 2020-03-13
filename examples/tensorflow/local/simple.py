"""
This script is a simple training script which uses Tensorflow's MonitoredSession interface.
It has been orchestrated with SageMaker Debugger hook to allow saving tensors during training.
Here, the hook has been created using its constructor to allow running this locally for your experimentation.
When you want to run this script in SageMaker, it is recommended to create the hook from json file.
Please see scripts in either 'sagemaker_byoc' or 'sagemaker_official_container' folder based on your use case.
"""

# Standard Library
import argparse
import random

# Third Party
import numpy as np
import tensorflow.compat.v1 as tf

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="S3 path for the model")
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
    parser.add_argument("--steps", type=int, help="Number of steps to run", default=100)
    parser.add_argument("--scale", type=float, help="Scaling factor for inputs", default=1.0)
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()

    # these random seeds are only intended for test purpose.
    # for now, 2,2,12 could promise no assert failure when running tests
    # if you wish to change the number, notice that certain steps' tensor value may be capable of variation
    if args.random_seed:
        tf.set_random_seed(2)
        np.random.seed(2)
        random.seed(12)

    hook = smd.EstimatorHook(
        out_dir=args.out_dir,
        include_collections=["weights", "gradients"],
        save_config=smd.SaveConfig(save_interval=args.save_interval),
    )

    # Network definition
    # Note the use of name scopes
    with tf.name_scope("foobar"):
        x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        w = tf.Variable(initial_value=[[10.0], [10.0]], name="weight1")
    with tf.name_scope("foobaz"):
        w0 = [[1], [1.0]]
        y = tf.matmul(x, w0)
    loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")

    hook.add_to_collection("losses", loss)

    global_step = tf.Variable(17, name="global_step", trainable=False)
    increment_global_step_op = tf.assign(global_step, global_step + 1)

    optimizer = tf.train.AdamOptimizer(args.lr)

    # Wrap the optimizer with wrap_optimizer so smdebug can find gradients to save
    optimizer = hook.wrap_optimizer(optimizer)

    # use this wrapped optimizer to minimize loss
    optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

    # pass the hook to hooks parameter of monitored session
    sess = tf.train.MonitoredSession(hooks=[hook])

    # use this session for running the tensorflow model
    hook.set_mode(smd.modes.TRAIN)
    for i in range(args.steps):
        x_ = np.random.random((10, 2)) * args.scale
        _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
        print(f"Step={i}, Loss={_loss}")

    hook.set_mode(smd.modes.EVAL)
    for i in range(args.steps):
        x_ = np.random.random((10, 2)) * args.scale
        sess.run([loss, increment_global_step_op], {x: x_})


if __name__ == "__main__":
    main()
