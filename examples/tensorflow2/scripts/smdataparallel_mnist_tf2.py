# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and limitations under the License.

# Third Party
import smdistributed.dataparallel.tensorflow as smdataparallel
import tensorflow as tf

# Register smdataparallel shutdown hook
smdataparallel.init()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[smdataparallel.local_rank()], "GPU")

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(
    path="mnist-%d.npz" % smdataparallel.rank()
)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32), tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
        tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
loss = tf.losses.SparseCategoricalCrossentropy()

opt = tf.optimizers.Adam(0.001 * smdataparallel.size())

checkpoint_dir = "/tmp/checkpoints"
checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # Create a new DistributedGradientTape, which uses TensorFlow’s GradientTape under the hood,
    # using an AllReduce to combine gradient values before applying gradients to model weights.
    tape = smdataparallel.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # Broadcast model and optimizer variable are first forward pass for sync
    if first_batch:
        smdataparallel.broadcast_variables(mnist_model.variables, root_rank=0)
        smdataparallel.broadcast_variables(opt.variables(), root_rank=0)

    return loss_value


for batch, (images, labels) in enumerate(dataset.take(1000 // smdataparallel.size())):
    loss_value = training_step(images, labels, batch == 0)

    if batch % 10 == 0 and smdataparallel.local_rank() == 0:
        print("Step #%d\tLoss: %.6f" % (batch, loss_value))

if smdataparallel.rank() == 0:
    checkpoint.save(checkpoint_dir)
