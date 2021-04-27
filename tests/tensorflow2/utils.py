# Standard Library
# Third Party
import tensorflow.compat.v2 as tf
from packaging import version

# Cached TF Version
TF_VERSION = version.parse(tf.__version__)


def is_tf_2_2():
    """
    TF 2.0 returns ['accuracy', 'batch', 'size'] as metric collections.
    where 'batch' is the batch number and size is the batch size.
    But TF 2.2 returns ['accuracy', 'batch'] in eager mode, reducing the total
    number of tensor_names emitted by 1.
    :return: bool
    """
    if TF_VERSION >= version.parse("2.2.0"):
        return True
    return False


def is_tf_2_3():
    if TF_VERSION == version.parse("2.3.0"):
        return True
    return False


def is_tf_version_greater_than_2_4_x():
    return version.parse("2.4.0") <= TF_VERSION


def _get_model():
    model = tf.keras.models.Sequential(
        [
            # WA for TF issue https://github.com/tensorflow/tensorflow/issues/36279
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def helper_keras_fit(trial_dir, hook):

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255

    model = _get_model()

    opt = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=False,
    )
    hooks = []
    hooks.append(hook)

    model.fit(x_train, y_train, epochs=1, steps_per_epoch=10, callbacks=hooks, verbose=0)
    model.evaluate(x_test, y_test, steps=10, callbacks=hooks, verbose=0)
    model.predict(x_test[:100], callbacks=hooks, verbose=0)

    model.save(trial_dir, save_format="tf")

    hook.close()


def helper_gradtape_tf(trial_dir, hook):
    def get_grads(images, labels):
        # with tf.GradientTape() as tape:
        return model(images, training=True)

    @tf.function
    def train_step(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(64)
    model = _get_model()
    opt = tf.keras.optimizers.Adam()

    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            labels = tf.one_hot(labels, depth=10)
            with tf.GradientTape() as tape:
                logits = train_step(data, labels)
            grads = tape.gradient(logits, model.variables)
            opt.apply_gradients(zip(grads, model.variables))

    model.save(trial_dir, save_format="tf")
    hook.close()
