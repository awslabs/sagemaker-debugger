# Third Party
import pytest


@pytest.fixture
def tf2_mnist_sequential_model():
    import tensorflow as tf

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


@pytest.fixture
def tf2_mnist_functional_model():
    import tensorflow as tf

    img_inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(img_inputs)
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)

    return tf.keras.models.Model(inputs=img_inputs, outputs=out)


@pytest.fixture
def tf2_mnist_subclassed_model():
    import tensorflow as tf

    class MyModel(tf.keras.models.Model):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(
                32,
                3,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12),
            )
            self.original_call = self.conv1.call

            def new_call(inputs, *args, **kwargs):
                # Since we use layer wrapper we need to assert if these parameters
                # are actually being passed into the original call fn
                assert kwargs["input_one"] == 1
                kwargs.pop("input_one")
                return self.original_call(inputs, *args, **kwargs)

            self.conv1.call = new_call
            self.conv0 = tf.keras.layers.Conv2D(
                32,
                3,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12),
            )
            self.flatten = tf.keras.layers.Flatten()
            self.d1 = tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=192),
            )
            self.d2 = tf.keras.layers.Dense(
                10, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=126)
            )
            self.bn = tf.keras.layers.BatchNormalization()

        def first(self, x):
            with tf.name_scope("first"):
                x = self.conv1(x, input_one=1)
                return self.flatten(x)

        def second(self, x):
            with tf.name_scope("second"):
                x = self.d1(x)
                return self.d2(x)

        def call(self, x, training=None):
            x = self.first(x)
            return self.second(x)

    return MyModel()


@pytest.fixture
def mnist_dataset():
    import tensorflow as tf

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train, y_train = x_train[:50000], y_train[:50000]
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(64)
    return dataset
