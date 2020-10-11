# Standard Library

# Third Party
import tensorflow as tf

assert tf.test.is_gpu_available()
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(path="mnist-%d.npz")

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32), tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
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
    opt = tf.optimizers.Adam()
    mnist_model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )


def lr_decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

callbacks = [lr_scheduler_callback]

# Train the model.
mnist_model.fit(dataset, steps_per_epoch=10, callbacks=callbacks, epochs=1)
