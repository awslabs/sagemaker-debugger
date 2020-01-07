"""
This script is a simple training script which uses Tensorflow's MonitoredSession interface.
It has been orchestrated with SageMaker Debugger hooks to allow saving tensors during training.
These hooks have been instrumented to read from json configuration that SageMaker will put in the training container.
Configuration provided to the SageMaker python SDK when creating a job will be passed on to the hook.
This allows you to use the same script with differing configurations across different runs.
If you use an official SageMaker Framework container (i.e. AWS Deep Learning Container), then
you do not have to orchestrate your script as below. Hooks will automatically be added in those environments.
For more information, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/
"""

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
parser.add_argument("--model_dir", type=str, help="S3 path for the model")
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
parser.add_argument("--steps", type=int, help="Number of steps to run", default=100)
parser.add_argument("--scale", type=float, help="Scaling factor for inputs", default=1.0)
parser.add_argument("--random_seed", type=bool, default=False)
args = parser.parse_args()

# these random seeds are only intended for test purpose.
# for now, 2,2,12 could promise no assert failure when running tests
# if you wish to change the number, notice that certain steps' tensor value may be capable of variation
if args.random_seed:
    tf.set_random_seed(2)
    np.random.seed(2)
    random.seed(12)

##### Enabling SageMaker Debugger ###########
# Create hook from the configuration provided through sagemaker python sdk
hook = smd.SessionHook.create_from_json_file()

# Network definition
# Note the use of name scopes
with tf.name_scope("foobar"):
    x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    w = tf.Variable(initial_value=[[10.0], [10.0]], name="weight1")
with tf.name_scope("foobaz"):
    w0 = [[1], [1.0]]
    y = tf.matmul(x, w0)
loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")

##### Enabling SageMaker Debugger ###########
# Adding custom loss to losses collection
hook.add_to_collection("losses", loss)

global_step = tf.Variable(17, name="global_step", trainable=False)
increment_global_step_op = tf.assign(global_step, global_step + 1)

optimizer = tf.train.AdamOptimizer(args.lr)

##### Enabling SageMaker Debugger ###########
# Wrap the optimizer with wrap_optimizer so smdebug can find gradients to save
optimizer = hook.wrap_optimizer(optimizer)

# use this wrapped optimizer to minimize loss
optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

##### Enabling SageMaker Debugger ###########
# pass the hook to hooks parameter of monitored session
sess = tf.train.MonitoredSession(hooks=[hook])

##### Enabling SageMaker Debugger ###########
# setting the mode of job so analysis can differentiate between TRAIN, EVAL, PREDICT
hook.set_mode(smd.modes.TRAIN)
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
    print(f"Step={i}, Loss={_loss}")

##### Enabling SageMaker Debugger ###########
# setting the mode of job so analysis can differentiate between TRAIN, EVAL, PREDICT
hook.set_mode(smd.modes.EVAL)
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    sess.run([loss, increment_global_step_op], {x: x_})
