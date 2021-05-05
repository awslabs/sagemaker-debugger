# Third Party
import pytest
import tensorflow.compat.v2 as tf
from packaging import version
from tests.utils import SagemakerSimulator

# First Party
import smdebug.tensorflow as smd

SMDEBUG_PREFIX = "smdebug_"


def get_keras_model_v2():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def get_keras_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (x_train, y_train), (x_test, y_test)


@pytest.fixture()
def tensorflow2_framework_override(monkeypatch):
    import smdebug.tensorflow.utils

    monkeypatch.setattr(smdebug.tensorflow.utils, "TF_VERSION", version.parse("1.14"))
    return


def test_tensorflow2_with_unsupported_version(
    tensorflow2_framework_override, script_mode: bool = False, eager_mode: bool = True
):
    """ Test the default ZCC behavior of saving losses and metrics in eager and non-eager modes."""
    smd.del_hook()
    tf.keras.backend.clear_session()
    run_eagerly = eager_mode
    with SagemakerSimulator() as sim:
        model = get_keras_model_v2()
        (x_train, y_train), (x_test, y_test) = get_keras_data()
        x_train, x_test = x_train / 255, x_test / 255

        opt = tf.keras.optimizers.RMSprop()
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
            run_eagerly=run_eagerly,
        )
        history = model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)
        test_scores = model.evaluate(x_test, y_test, verbose=2)

        hook = smd.get_hook()
        assert hook == None
