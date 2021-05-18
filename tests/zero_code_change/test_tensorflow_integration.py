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
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from tests.constants import TEST_DATASET_S3_PATH
from tests.tensorflow.hooks.test_mirrored_strategy import test_basic
from tests.tensorflow.keras.test_keras_mirrored import test_tf_keras
from tests.utils import use_s3_datasets
from tests.zero_code_change.tf_utils import (
    get_data,
    get_estimator,
    get_input_fns,
    get_keras_data,
    get_keras_model_v1,
    get_train_op_and_placeholders,
)

# First Party
import smdebug.tensorflow as smd
from smdebug.core.utils import SagemakerSimulator


def helper_train(script_mode=False, sim=None, train_steps=80, eval_steps=20):
    # Setup
    mnist_classifier = get_estimator()
    train_input_fn, eval_input_fn = get_input_fns()

    # Train and evaluate

    if script_mode:
        hook = smd.EstimatorHook(out_dir=sim.out_dir)
        hook.set_mode(mode=smd.modes.TRAIN)
        mnist_classifier.train(input_fn=train_input_fn, steps=train_steps, hooks=[hook])
        hook.set_mode(mode=smd.modes.EVAL)
        mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps, hooks=[hook])
    else:
        mnist_classifier.train(input_fn=train_input_fn, steps=train_steps)
        mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps)


@pytest.mark.parametrize("script_mode", [False])
def test_estimator(script_mode):
    """ Works as intended. """
    smd.del_hook()
    tf.reset_default_graph()
    with SagemakerSimulator() as sim:
        train_steps, eval_steps = 80, 20
        helper_train(
            script_mode=script_mode, sim=sim, train_steps=train_steps, eval_steps=eval_steps
        )

        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        print(trial)
        assert smd.get_hook() is not None, "Hook was not created."
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."
        assert trial.steps() == [0, train_steps], "Wrong step count for trial."


def helper_test_estimator_gradients_zcc(nested=False, mirrored=False):
    """ Works as intended. """
    smd.del_hook()
    tf.reset_default_graph()
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
                }
            ]
        }
        """
    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:

        if mirrored:
            test_basic("/opt/ml/output/tensors", zcc=True)
        else:
            # Setup
            mnist_classifier = get_estimator(nested_optimizer=nested, mirrored=mirrored)
            train_input_fn, eval_input_fn = get_input_fns()

            # Train and evaluate
            train_steps, eval_steps = 10, 10
            mnist_classifier.train(input_fn=train_input_fn, steps=train_steps)
            mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps)

            # Check that hook created and tensors saved
            trial = smd.create_trial(path=sim.out_dir)
            print(trial)
            assert smd.get_hook() is not None, "Hook was not created."
            assert len(trial.steps()) > 0, "Nothing saved at any step."
            assert len(trial.tensor_names()) > 0, "Tensors were not saved."
            assert trial.steps() == [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
            ], "Wrong step count for trial."
            print(trial.tensor_names(collection="gradients"))
            assert len(trial.tensor_names(collection="gradients")) > 0
            assert len(trial.tensor_names(collection="weights")) > 0
            assert len(trial.tensor_names(collection="losses")) > 0
            assert len(trial.tensor(trial.tensor_names(collection="gradients")[0]).steps()) > 0
            assert len(trial.modes()) == 2


def test_estimator_gradients_zcc_nested():
    helper_test_estimator_gradients_zcc(nested=True)


def test_estimator_gradients_zcc_mirrored():
    helper_test_estimator_gradients_zcc(nested=False, mirrored=True)


@pytest.mark.parametrize("script_mode", [False])
def test_linear_classifier(script_mode):
    """ Works as intended. """
    smd.del_hook()
    tf.reset_default_graph()
    with SagemakerSimulator() as sim:
        # Setup
        train_input_fn, eval_input_fn = get_input_fns()
        x_feature = tf.feature_column.numeric_column("x", shape=(28, 28))
        estimator = tf.estimator.LinearClassifier(
            feature_columns=[x_feature], model_dir="/tmp/mnist_linear_classifier", n_classes=10
        )

        # Train
        if script_mode:
            hook = smd.EstimatorHook(out_dir=sim.out_dir)
            estimator.train(input_fn=train_input_fn, steps=100, hooks=[hook])
        else:
            estimator.train(input_fn=train_input_fn, steps=100)

        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert smd.get_hook() is not None, "Hook was not created."
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."


@pytest.mark.parametrize("script_mode", [False])
def test_monitored_session(script_mode):
    """ Works as intended. """
    smd.del_hook()
    tf.reset_default_graph()
    json_file_contents = """
            {
                "S3OutputPath": "s3://sagemaker-test",
                "LocalPath": "/opt/ml/output/tensors",
                "HookParameters" : {
                    "save_interval": "100"
                }
            }
            """
    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
        train_op, X, Y = get_train_op_and_placeholders()
        init = tf.global_variables_initializer()
        mnist = get_data()

        if script_mode:
            hook = smd.SessionHook(out_dir=sim.out_dir)
            sess = tf.train.MonitoredSession(hooks=[hook])
        else:
            sess = tf.train.MonitoredSession()

        with sess:
            sess.run(init)
            for step in range(1, 101):
                batch_x, batch_y = mnist.train.next_batch(32)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert smd.get_hook() is not None, "Hook was not created."
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."


def test_monitored_session_gradients_zcc():
    """ Works as intended. """
    smd.del_hook()
    json_file_contents = """
    {
        "S3OutputPath": "s3://sagemaker-test",
        "LocalPath": "/opt/ml/output/tensors",
        "HookParameters" : {
            "save_interval": "100"
        },
        "CollectionConfigurations": [
            {
                "CollectionName": "gradients"
            },
            {
                "CollectionName": "losses"
            }
        ]
    }
    """
    tf.reset_default_graph()
    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
        train_op, X, Y = get_train_op_and_placeholders()
        init = tf.global_variables_initializer()
        mnist = get_data()

        sess = tf.train.MonitoredSession()

        with sess:
            sess.run(init)
            for step in range(1, 101):
                batch_x, batch_y = mnist.train.next_batch(32)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert smd.get_hook() is not None, "Hook was not created."
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."
        assert len(trial.tensor_names(collection="gradients")) > 0


@pytest.mark.parametrize("script_mode", [False])
def test_keras_v1(script_mode):
    """ Works as intended. """
    smd.del_hook()
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    with SagemakerSimulator() as sim:
        model = get_keras_model_v1()
        (x_train, y_train), (x_test, y_test) = get_keras_data()

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )
        if script_mode:
            hook = smd.KerasHook(out_dir=sim.out_dir)
            history = model.fit(
                x_train, y_train, batch_size=64, epochs=5, validation_split=0.2, callbacks=[hook]
            )
            test_scores = model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])
        else:
            history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
            test_scores = model.evaluate(x_test, y_test, verbose=2)

        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert smd.get_hook() is not None, "Hook was not created."
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."


@pytest.mark.parametrize("script_mode", [False])
@pytest.mark.parametrize("tf_optimizer", [True, False])
def test_keras_gradients(script_mode, tf_optimizer):
    """ Works as intended. """
    smd.del_hook()
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    json_file_contents = """
            {
                "S3OutputPath": "s3://sagemaker-test",
                "LocalPath": "/opt/ml/output/tensors",
                "CollectionConfigurations": [
                    {
                        "CollectionName": "gradients"
                    },
                    {
                        "CollectionName": "optimizer_variables"
                    },
                    {
                        "CollectionName": "losses"
                    }
                ]
            }
            """
    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
        model = get_keras_model_v1()
        (x_train, y_train), (x_test, y_test) = get_keras_data()

        if tf_optimizer:
            opt = tf.train.RMSPropOptimizer(0.1)
        else:
            opt = tf.keras.optimizers.RMSprop()

        if script_mode:
            hook = smd.KerasHook(
                out_dir=sim.out_dir,
                include_collections=["gradients", "optimizer_variables", "losses"],
            )
            opt = hook.wrap_optimizer(opt)
            model.compile(
                loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )
            history = model.fit(
                x_train, y_train, batch_size=16, epochs=5, validation_split=0.2, callbacks=[hook]
            )
            test_scores = model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])
        else:
            model.compile(
                loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )
            history = model.fit(x_train, y_train, batch_size=16, epochs=5, validation_split=0.2)
            test_scores = model.evaluate(x_test, y_test, verbose=2)

        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        assert smd.get_hook() is not None, "Hook was not created."
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensor_names()) > 0, "Tensors were not saved."
        assert len(trial.tensor_names(collection="gradients")) > 0
        if not tf_optimizer:
            # as this is only supported for keras optimizers currently
            assert len(trial.tensor_names(collection="optimizer_variables")) > 0


@pytest.mark.parametrize("script_mode", [False])
def test_keras_gradients_tf_opt(script_mode):
    test_keras_gradients(script_mode=script_mode, tf_optimizer=True)


def test_keras_gradients_mirrored(include_workers="one"):
    """ Works as intended. """
    smd.del_hook()
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    json_file_contents_p1 = """
            {
                "S3OutputPath": "s3://sagemaker-test",
                "LocalPath": "/opt/ml/output/tensors",
                "HookParameters" : {

            """
    json_file_contents_p2 = f'"include_workers": "{include_workers}",'
    json_file_contents_p3 = """
                    "save_interval": "3"
                },
                "CollectionConfigurations": [
                    {
                        "CollectionName": "gradients"
                    },
                    {
                        "CollectionName": "optimizer_variables"
                    },
                    {
                        "CollectionName": "losses"
                    },
                    {
                        "CollectionName": "weights"
                    },
                    {
                        "CollectionName": "biases"
                    },
                    {
                        "CollectionName": "outputs"
                    },
                    {
                        "CollectionName": "metrics"
                    }
                ]
            }
            """
    json_file_contents = json_file_contents_p1 + json_file_contents_p2 + json_file_contents_p3
    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
        test_tf_keras("/opt/ml/output/tensors", zcc=True, include_workers=include_workers)


def test_keras_gradients_mirrored_all_workers():
    test_keras_gradients_mirrored(include_workers="all")


@pytest.mark.parametrize("script_mode", [False])
def test_keras_to_estimator(script_mode):
    """ Works as intended. """
    import tensorflow.compat.v1.keras as keras

    tf.reset_default_graph()
    smd.del_hook()
    keras.backend.clear_session()
    with SagemakerSimulator() as sim:
        model = keras.models.Sequential(
            [
                keras.layers.Dense(16, activation="relu", input_shape=(4,)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        def input_fn():
            split = tfds.Split.TRAIN
            data_dir = TEST_DATASET_S3_PATH if use_s3_datasets() else None
            dataset = tfds.load("iris", data_dir=data_dir, split=split, as_supervised=True)
            dataset = dataset.map(lambda features, labels: ({"dense_input": features}, labels))
            dataset = dataset.batch(32).repeat()
            return dataset

        model.compile(loss="categorical_crossentropy", optimizer="adam")
        model.summary()

        keras_estimator = tf.keras.estimator.model_to_estimator(
            keras_model=model, model_dir=sim.out_dir
        )

        if script_mode:
            hook = smd.EstimatorHook(sim.out_dir)
            hook.set_mode(smd.modes.TRAIN)
            keras_estimator.train(input_fn=input_fn, steps=25, hooks=[hook])
            hook.set_mode(smd.modes.EVAL)
            eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10, hooks=[hook])
        else:
            keras_estimator.train(input_fn=input_fn, steps=25)
            keras_estimator.evaluate(input_fn=input_fn, steps=10)

        tr = smd.create_trial(sim.out_dir)
        assert len(tr.tensor_names()) == 1
        assert tr.steps() == [0, 25]
        assert len(tr.steps(smd.modes.TRAIN)) == 1
        assert len(tr.steps(smd.modes.EVAL)) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-mode", help="Manually create hooks instead of relying on ZCC", action="store_true"
    )
    args = parser.parse_args()
    script_mode_arg = args.script_mode

    test_monitored_session(script_mode=script_mode_arg)
    if not script_mode_arg:
        test_monitored_session_gradients_zcc()
    test_estimator(script_mode=script_mode_arg)
    if not script_mode_arg:
        helper_test_estimator_gradients_zcc()
        test_estimator_gradients_zcc_nested()
        test_estimator_gradients_zcc_mirrored()
    test_linear_classifier(script_mode=script_mode_arg)
    test_keras_v1(script_mode=script_mode_arg)
    test_keras_gradients(script_mode=script_mode_arg)
    test_keras_gradients_tf_opt(script_mode=script_mode_arg)
    test_keras_to_estimator(script_mode=script_mode_arg)
    if not script_mode_arg:
        test_keras_gradients_mirrored_all_workers()
        test_keras_gradients_mirrored()
