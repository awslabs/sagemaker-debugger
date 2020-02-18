"""
This file is temporary, for testing with 2.X.
We'll need to integrate a more robust testing pipeline and make this part of pytest
before pushing to master.

This was tested with TensorFlow 2.1, by running
`python tests/tensorflow2/test_simple.py` from the main directory.
"""

# Standard Library
from tempfile import TemporaryDirectory

# Third Party
import pytest
import tensorflow.compat.v2 as tf

# First Party
import smdebug.tensorflow as smd


def helper_keras_fit(eager=True, saveall=True):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    with TemporaryDirectory() as dirpath:
        hook = smd.KerasHook(out_dir=dirpath, save_all=saveall)

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                      experimental_run_tf_function=eager)
        model.fit(x_train, y_train, epochs=1, callbacks=[hook])
        model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])

        trial = smd.create_trial(path=dirpath)
        print(hook)
        print(trial)


@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("saveall", [True, False])
def test_keras_fit(eager, saveall):
    helper_keras_fit(eager, saveall)
