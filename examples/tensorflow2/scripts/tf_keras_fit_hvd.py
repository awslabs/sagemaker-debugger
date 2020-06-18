# Standard Library
import argparse
from datetime import datetime

# Third Party
import horovod.tensorflow.keras as hvd
import tensorflow.compat.v2 as tf


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_data(batch_size):
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(
        path="mnist-%d.npz" % hvd.rank()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64),
        )
    )
    dataset = dataset.repeat().shuffle(10000).batch(batch_size)
    return dataset


def get_model():
    mnist_model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, [3, 3], activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return mnist_model


def train(model, dataset, epoch):
    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )

    # Create a TensorBoard callback
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs, histogram_freq=1, profile_batch=2
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
        tboard_callback,
    ]
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint("checkpoint-{epoch}.h5"))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    model.fit(
        dataset,
        steps_per_epoch=500 // hvd.size(),
        callbacks=callbacks,
        epochs=epoch,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="Tensorflow2 MNIST Example")
    parser.add_argument("--use_only_cpu", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--model_dir", type=str, default="/tmp/mnist_model")

    args = parser.parse_args()

    # constants
    lr = 0.001
    batch_size = 64

    # Horovod: initialize library.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(lr * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt)

    dataset = get_data(batch_size)
    mnist_model = get_model()

    train(model=mnist_model, dataset=dataset, epoch=args.num_epochs)
