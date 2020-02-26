"""
WARNING: This must be run manually, with the custom TensorFlow fork installed.
Not used in CI/CD. May be useful for DLC testing.

Be sure to run with Python2 (/usr/bin/python) and Python3.
Run with and without the flag --zcc.

Test with DNNClassifier and raw Estimator.
Test with Session.
Test with Keras.

Test with AdamOptimizer and SGD.

We check that certain tensors are saved.
Here in the test suite we delete the hook after every script.
"""
# Standard Library
import argparse

# Third Party
import tensorflow.compat.v2 as tf

# First Party
import smdebug.tensorflow as smd
from smdebug.core.utils import SagemakerSimulator


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
    x_train, x_test = x_train / 255, x_test / 255

    return (x_train, y_train), (x_test, y_test)


def test_keras_v2(script_mode: bool = False, eager_mode: bool = True):
    """ Works as intended. """
    smd.del_hook()
    with SagemakerSimulator() as sim:
        model = get_keras_model_v2()
        (x_train, y_train), (x_test, y_test) = get_keras_data()

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=["accuracy"],
            experimental_run_tf_function=eager_mode,
        )
        if script_mode:
            hook = smd.KerasHook(out_dir=sim.out_dir, export_tensorboard=True)
            history = model.fit(
                x_train, y_train, batch_size=64, epochs=5, validation_split=0.2, callbacks=[hook]
            )
            test_scores = model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])
        else:
            history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
            test_scores = model.evaluate(x_test, y_test, verbose=2)

        hook = smd.get_hook()
        assert hook
        hook.close()
        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-mode", help="Manually create hooks instead of relying on ZCC", action="store_true"
    )
    args = parser.parse_args()
    script_mode = args.script_mode

    # eager mode
    test_keras_v2(script_mode=script_mode)
    # non-eager mode
    test_keras_v2(script_mode=script_mode, eager_mode=False)
