# Standard Library

# Third Party
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

assert tf.test.is_gpu_available()

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

train_dataset, valid_dataset = tfds.load(
    "cifar10", split=["train", "test"], batch_size=8, as_supervised=True
)

train_dataset = train_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

valid_dataset = valid_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
valid_dataset = valid_dataset.repeat()


def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation="relu", padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation("relu")(x)
    return x


def non_res_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation="relu", padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    return x


inputs = keras.Input(shape=(24, 24, 3))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)

num_res_net_blocks = 10
for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)

x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)

res_net_model = keras.Model(inputs, outputs)

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.Adam(0.001 * hvd.size())

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

res_net_model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["acc"],
    experimental_run_tf_function=False,
)


callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
]

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

res_net_model.fit(
    train_dataset,
    epochs=2,
    steps_per_epoch=350,
    validation_data=valid_dataset,
    validation_steps=25,
    callbacks=callbacks,
    verbose=verbose,
)
