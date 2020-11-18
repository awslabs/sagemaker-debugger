# Third Party
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model
from tests.tensorflow2.utils import is_tf_2_2

# First Party
import smdebug.tensorflow as smd


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(
            32, 3, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12)
        )
        self.original_call = self.conv1.call

        def new_call(inputs, *args, **kwargs):
            # Since we use layer wrapper we need to assert if these parameters
            # are actually being passed into the original call fn
            assert kwargs["input_one"] == 1
            kwargs.pop("input_one")
            return self.original_call(inputs, *args, **kwargs)

        self.conv1.call = new_call
        self.conv0 = Conv2D(
            32, 3, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12)
        )
        self.flatten = Flatten()
        self.d1 = Dense(
            128, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=192)
        )
        self.d2 = Dense(10, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=126))
        self.bn = BatchNormalization()

    def first(self, x):
        with tf.name_scope("first"):
            tf.print("mymodel.first")
            x = self.conv1(x, input_one=1)
            return self.flatten(x)

    def second(self, x):
        with tf.name_scope("second"):
            x = self.d1(x)
            return self.d2(x)

    def call(self, x, training=None):
        x = self.first(x)
        return self.second(x)


def test_subclassed_model(out_dir):
    # Download and load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data("MNIST-data")
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Create an instance of the model
    model = MyModel()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000, seed=123).batch(2)
    )

    MyModel.hook = smd.KerasHook(
        out_dir,
        save_all=True,
        save_config=smd.SaveConfig(save_steps=[x for x in range(10)], save_interval=1),
    )

    MyModel.hook.register_model(model)
    model.compile(optimizer="Adam", loss="mse", run_eagerly=True)
    model.fit(train_ds, epochs=1, steps_per_epoch=10, callbacks=[MyModel.hook])

    trial = smd.create_trial(out_dir)
    assert len(trial.tensor_names(collection=smd.CollectionKeys.LAYERS)) == 8

    assert trial.tensor_names(collection=smd.CollectionKeys.LOSSES) == ["loss"]
    if is_tf_2_2():
        # Feature to save model inputs and outputs was first added for TF 2.2.0
        assert trial.tensor_names(collection=smd.CollectionKeys.INPUTS) == ["model_input"]
        assert trial.tensor_names(collection=smd.CollectionKeys.OUTPUTS) == [
            "labels",
            "predictions",
        ]
        assert len(trial.tensor_names(collection=smd.CollectionKeys.GRADIENTS)) == 6
