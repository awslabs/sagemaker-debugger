# Future
from __future__ import absolute_import, division, print_function, unicode_literals

# Third Party
import tensorflow as tf
import tensorflow_datasets as tfds

# First Party
from smdebug.core.collection import CollectionKeys
from smdebug.tensorflow import KerasHook, get_collection

tfds.disable_progress_bar()


def train_model():
    print(tf.__version__)

    datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)

    mnist_train, mnist_test = datasets["train"], datasets["test"]

    strategy = tf.distribute.MirroredStrategy()

    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    num_train_examples = info.splits["train"].num_examples
    num_test_examples = info.splits["test"].num_examples

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    hook = KerasHook(
        out_dir="~/ts_outputs/",
        include_collections=[
            # CollectionKeys.WEIGHTS,
            # CollectionKeys.GRADIENTS,
            # CollectionKeys.OPTIMIZER_VARIABLES,
            CollectionKeys.DEFAULT,
            # CollectionKeys.METRICS,
            # CollectionKeys.LOSSES,
            # CollectionKeys.OUTPUTS,
            # CollectionKeys.SCALARS,
        ],
        save_all=True,
    )

    with strategy.scope():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=hook.wrap_optimizer(tf.keras.optimizers.Adam()),
            metrics=["accuracy"],
        )

    # get_collection('default').include('Relu')

    callbacks = [
        hook
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    model.fit(train_dataset, epochs=1, callbacks=callbacks)
    model.predict(eval_dataset, callbacks=callbacks)
    model.fit(train_dataset, epochs=1, callbacks=callbacks)
