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

    return (x_train, y_train), (x_test, y_test)


def helper_test_keras_v2(script_mode: bool = False, eager_mode: bool = True):
    """ Test the default ZCC behavior of saving losses and metrics in eager and non-eager modes."""
    smd.del_hook()
    tf.keras.backend.clear_session()
    if not eager_mode:
        tf.compat.v1.disable_eager_execution()
    with SagemakerSimulator() as sim:
        model = get_keras_model_v2()
        (x_train, y_train), (x_test, y_test) = get_keras_data()
        x_train, x_test = x_train / 255, x_test / 255

        opt = tf.keras.optimizers.RMSprop()
        if script_mode:
            hook = smd.KerasHook(out_dir=sim.out_dir, export_tensorboard=True)
            opt = hook.wrap_optimizer(opt)
            model.compile(
                loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )
            history = model.fit(
                x_train, y_train, batch_size=64, epochs=2, validation_split=0.2, callbacks=[hook]
            )
            test_scores = model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])
        else:
            model.compile(
                loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )
            history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
            test_scores = model.evaluate(x_test, y_test, verbose=2)

        hook = smd.get_hook()
        assert hook
        hook.close()
        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."
        assert len(trial.tensor_names(collection="losses")) > 0


def helper_test_keras_v2_json_config(
    json_file_contents, script_mode: bool = False, eager_mode: bool = True
):
    """ Tests ZCC with custom hook configs """
    smd.del_hook()
    tf.keras.backend.clear_session()
    if not eager_mode:
        tf.compat.v1.disable_eager_execution()
    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
        model = get_keras_model_v2()
        (x_train, y_train), (x_test, y_test) = get_keras_data()
        x_train, x_test = x_train / 255, x_test / 255

        opt = tf.keras.optimizers.RMSprop()
        if script_mode:
            hook = smd.KerasHook.create_from_json_file()
            opt = hook.wrap_optimizer(opt)
            model.compile(
                loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )
            history = model.fit(
                x_train, y_train, batch_size=64, epochs=2, validation_split=0.2, callbacks=[hook]
            )
            test_scores = model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])
        else:
            model.compile(
                loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )
            history = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.2)
            test_scores = model.evaluate(x_test, y_test, verbose=2)

        hook = smd.get_hook()
        assert hook
        hook.close()
        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."
        if not eager_mode:
            assert len(trial.tensor_names(collection="gradients")) > 0
        assert len(trial.tensor_names(collection="weights")) > 0
        assert len(trial.tensor_names(collection="losses")) > 0


def test_keras_v2_default(script_mode: bool = False, eager_mode: bool = True):
    # Test default ZCC behavior
    helper_test_keras_v2(script_mode=script_mode, eager_mode=eager_mode)


def test_keras_v2_multi_collections(script_mode: bool = False, eager_mode: bool = True):
    # Test multiple collections included in hook json
    json_file_contents = """
            {
                "S3OutputPath": "s3://sagemaker-test",
                "LocalPath": "/opt/ml/output/tensors",
                "HookParameters" : {
                    "save_interval": "2",
                    "include_workers": "all"
                },
                "CollectionConfigurations": [
                    {
                        "CollectionName": "gradients"
                    },
                    {
                        "CollectionName": "weights"
                    },
                    {
                        "CollectionName": "losses"
                    },
                    {
                        "CollectionName": "biases"
                    },
                    {
                        "CollectionName": "optimizer_variables"
                    }
                ]
            }
            """
    helper_test_keras_v2_json_config(
        script_mode=script_mode, eager_mode=eager_mode, json_file_contents=json_file_contents
    )


def test_keras_v2_save_all(script_mode: bool = False, eager_mode: bool = True):
    # Test save all through hook config
    json_file_contents = """
            {
                "S3OutputPath": "s3://sagemaker-test",
                "LocalPath": "/opt/ml/output/tensors",
                "HookParameters" : {
                    "save_steps": "0,1,2,3",
                    "save_all": true
                }
            }
            """
    helper_test_keras_v2_json_config(
        script_mode=script_mode, eager_mode=eager_mode, json_file_contents=json_file_contents
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-mode", help="Manually create hooks instead of relying on ZCC", action="store_true"
    )
    args = parser.parse_args()
    script_mode = args.script_mode

    # eager mode
    test_keras_v2_default(script_mode)
    test_keras_v2_multi_collections(script_mode)
    test_keras_v2_save_all(script_mode)

    # non-eager mode
    test_keras_v2_default(script_mode, eager_mode=False)
    test_keras_v2_multi_collections(script_mode, eager_mode=False)
    test_keras_v2_save_all(script_mode, eager_mode=False)
