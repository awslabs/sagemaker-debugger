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
import pytest
import tensorflow.compat.v2 as tf
from tests.tensorflow2.utils import is_tf_2_3
from tests.utils import SagemakerSimulator

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys


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
    if not eager_mode and is_tf_2_3() is False:
        # v1 training APIs are currently not supported
        # in ZCC mode with smdebug 0.9 and AWS TF 2.3.0
        tf.compat.v1.disable_eager_execution()

    # Performance regression in the _make_histogram fn
    enable_tb = False if tf.__version__ == "2.0.2" or is_tf_2_3() else True
    with SagemakerSimulator(enable_tb=enable_tb) as sim:
        model = get_keras_model_v2()
        (x_train, y_train), (x_test, y_test) = get_keras_data()
        x_train, x_test = x_train / 255, x_test / 255
        run_eagerly = None
        if is_tf_2_3():
            # Test eager and non eager mode for v2
            run_eagerly = eager_mode

        opt = tf.keras.optimizers.RMSprop()
        if script_mode:
            hook = smd.KerasHook(out_dir=sim.out_dir, export_tensorboard=True)
            opt = hook.wrap_optimizer(opt)
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"],
                run_eagerly=run_eagerly,
            )
            history = model.fit(
                x_train, y_train, batch_size=64, epochs=2, validation_split=0.2, callbacks=[hook]
            )
            test_scores = model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])
        else:
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"],
                run_eagerly=run_eagerly,
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

        # DEFAULT TENSORS SAVED
        assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) > 0, "No Losses Saved"
        assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) > 0, "No Metrics Saved"
        assert (
            len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 0
        ), "Weights were not expected to be saved by default"
        assert (
            len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
        ), "Biases were not expected to be saved by default"


def helper_test_keras_v2_json_config(
    json_file_contents, script_mode: bool = False, eager_mode: bool = True
):
    """ Tests ZCC with custom hook configs """
    smd.del_hook()
    tf.keras.backend.clear_session()
    if not eager_mode and is_tf_2_3() is False:
        # v1 training APIs are currently not supported
        # in ZCC mode with smdebug 0.9 and AWS TF 2.3.0
        tf.compat.v1.disable_eager_execution()

    # Performance regression in the _make_histogram fn
    enable_tb = False if tf.__version__ == "2.0.2" or is_tf_2_3() else True

    with SagemakerSimulator(json_file_contents=json_file_contents, enable_tb=enable_tb) as sim:
        model = get_keras_model_v2()
        (x_train, y_train), (x_test, y_test) = get_keras_data()
        x_train, x_test = x_train / 255, x_test / 255

        opt = tf.keras.optimizers.RMSprop()
        run_eagerly = None
        if is_tf_2_3():
            # Test eager and non eager mode for v2
            run_eagerly = eager_mode
        if script_mode:
            hook = smd.KerasHook.create_from_json_file()
            opt = hook.wrap_optimizer(opt)
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"],
                run_eagerly=run_eagerly,
            )
            history = model.fit(
                x_train, y_train, batch_size=64, epochs=2, validation_split=0.2, callbacks=[hook]
            )
            test_scores = model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])
        else:
            model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["accuracy"],
                run_eagerly=run_eagerly,
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
        if not eager_mode and is_tf_2_3() is False:
            # Gradients are currently not saved in ZCC mode with AWS TF 2.3.0
            # and smdebug 0.9
            assert len(trial.tensor_names(collection="gradients")) > 0
        assert len(trial.tensor_names(collection="weights")) > 0
        assert len(trial.tensor_names(collection="losses")) > 0


@pytest.mark.parametrize("script_mode", [False])
@pytest.mark.parametrize("eager_mode", [True, False])
def test_keras_v2_default(script_mode, eager_mode):
    # Test default ZCC behavior
    helper_test_keras_v2(script_mode=script_mode, eager_mode=eager_mode)


@pytest.mark.parametrize("script_mode", [False])
@pytest.mark.parametrize("eager_mode", [True, False])
def test_keras_v2_multi_collections(script_mode, eager_mode):
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


@pytest.mark.parametrize("script_mode", [False])
@pytest.mark.parametrize("eager_mode", [True, False])
def test_keras_v2_save_all(script_mode, eager_mode):
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
        "--script-mode",
        help="Manually create hooks instead of relying on ZCC",
        action="store_true",
        default=False,
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
