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
import tensorflow as tf
import tensorflow_datasets as tfds
from tf_utils import (
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


def test_estimator(script_mode: bool):
    """ Works as intended. """
    smd.del_hook()
    with SagemakerSimulator() as sim:
        # Setup
        mnist_classifier = get_estimator()
        train_input_fn, eval_input_fn = get_input_fns()

        # Train and evaluate
        train_steps, eval_steps = 80, 20
        if script_mode:
            hook = smd.EstimatorHook(out_dir=sim.out_dir)
            mnist_classifier.train(input_fn=train_input_fn, steps=train_steps, hooks=[hook])
            mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps, hooks=[hook])
        else:
            mnist_classifier.train(input_fn=train_input_fn, steps=train_steps)
            mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        # Check that hook created and tensors saved
        trial = smd.create_trial(path=sim.out_dir)
        print(trial)
        assert smd.get_hook() is not None, "Hook was not created."
        assert len(trial.steps()) > 0, "Nothing saved at any step."
        assert len(trial.tensors()) > 0, "Tensors were not saved."
        assert trial.steps() == [0, train_steps], "Wrong step count for trial."


def test_linear_classifier(script_mode: bool):
    """ Works as intended. """
    smd.del_hook()
    with SagemakerSimulator() as sim:
        # Setup
        train_input_fn, eval_input_fn = get_input_fns()
        x_feature = tf.feature_column.numeric_column("x", shape=(28, 28))
        estimator = tf.compat.v1.estimator.LinearClassifier(
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
        assert len(trial.tensors()) > 0, "Tensors were not saved."


def test_monitored_session(script_mode: bool):
    """ Works as intended. """
    smd.del_hook()
    with SagemakerSimulator() as sim:
        train_op, X, Y = get_train_op_and_placeholders()
        init = tf.compat.v1.global_variables_initializer()
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
        assert len(trial.tensors()) > 0, "Tensors were not saved."


def test_keras_v1(script_mode: bool):
    """ Works as intended. """
    smd.del_hook()
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
        assert len(trial.tensors()) > 0, "Tensors were not saved."


def test_keras_to_estimator(script_mode: bool):
    """ Works as intended. """
    import tensorflow.compat.v1.keras as keras

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
            dataset = tfds.load("iris", split=split, as_supervised=True)
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
        assert len(tr.tensors()) == 1
        assert tr.steps() == [0, 25]
        assert len(tr.steps(smd.modes.TRAIN)) == 1
        assert len(tr.steps(smd.modes.EVAL)) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-mode", help="Manually create hooks instead of relying on ZCC", action="store_true"
    )
    args = parser.parse_args()
    script_mode = args.script_mode

    test_monitored_session(script_mode=script_mode)
    test_estimator(script_mode=script_mode)
    test_linear_classifier(script_mode=script_mode)
    test_keras_v1(script_mode=script_mode)
    test_keras_to_estimator(script_mode=script_mode)
