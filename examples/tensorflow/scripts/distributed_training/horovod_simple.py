#  Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

# Future
from __future__ import absolute_import, division, print_function

# Standard Library
import argparse

# Third Party
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf

# First Party
import smdebug.tensorflow as smd

tf.logging.set_verbosity(tf.logging.INFO)


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
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
parser.add_argument("--steps", type=int, help="Number of steps to run", default=100)
parser.add_argument("--scale", type=float, help="Scaling factor for inputs", default=1.0)
parser.add_argument("--save_all", type=str2bool, default=True)
parser.add_argument("--save_path", type=str, default="/opt/ml/output/tensors")
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

# Wrap with Horovod Distributed Optimizer.
optimizer = hvd.DistributedOptimizer(optimizer)

# Wrap the optimizer with wrap_optimizer so Sagemaker debugger can find gradients and optimizer_variables to save
optimizer = hook.wrap_optimizer(optimizer)

# use this wrapped optimizer to minimize loss
optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

# Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
# rank 0 to all other processes. This is necessary to ensure consistent
# initialization of all workers when training is started with random weights or
# restored from a checkpoint.
bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

hook.set_mode(smd.modes.TRAIN)

# pass the hook to hooks parameter of monitored session
sess = tf.train.MonitoredSession(hooks=[hook])

# use this session for running the tensorflow model
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
    print(f"Step={i}, Loss={_loss}")

hook.set_mode(smd.modes.EVAL)
for i in range(args.steps):
    x_ = np.random.random((10, 2)) * args.scale
    sess.run([loss, increment_global_step_op], {x: x_})
