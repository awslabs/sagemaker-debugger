# Standard Library
# Third Party
import pytest
import tensorflow.compat.v2 as tf
from tests.zero_code_change.tf_utils import get_estimator, get_input_fns

# First Party
import smdebug.tensorflow as smd


@pytest.mark.parametrize("saveall", [True, False])
def test_estimator(out_dir, tf_eager_mode, saveall):
    """ Works as intended. """
    if tf_eager_mode is False:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    mnist_classifier = get_estimator()
    train_input_fn, eval_input_fn = get_input_fns()

    # Train and evaluate
    train_steps, eval_steps = 80, 20
    hook = smd.EstimatorHook(out_dir=out_dir, save_all=saveall)
    hook.set_mode(mode=smd.modes.TRAIN)
    mnist_classifier.train(input_fn=train_input_fn, steps=train_steps, hooks=[hook])
    hook.set_mode(mode=smd.modes.EVAL)
    mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps, hooks=[hook])

    # Check that hook created and tensors saved
    trial = smd.create_trial(path=out_dir)
    assert smd.get_hook() is not None, "Hook was not created."
    assert len(trial.steps()) > 0, "Nothing saved at any step."
    assert len(trial.tensor_names()) > 0, "Tensors were not saved."
    assert trial.steps() == [0, train_steps], "Wrong step count for trial."


@pytest.mark.parametrize("saveall", [True, False])
def test_linear_classifier(out_dir, tf_eager_mode, saveall):
    """ Works as intended. """
    if tf_eager_mode is False:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    train_input_fn, eval_input_fn = get_input_fns()
    x_feature = tf.feature_column.numeric_column("x", shape=(28, 28))
    estimator = tf.estimator.LinearClassifier(
        feature_columns=[x_feature], model_dir="/tmp/mnist_linear_classifier", n_classes=10
    )
    hook = smd.EstimatorHook(out_dir=out_dir, save_all=saveall)
    estimator.train(input_fn=train_input_fn, steps=100, hooks=[hook])

    # Check that hook created and tensors saved
    trial = smd.create_trial(path=out_dir)
    assert smd.get_hook() is not None, "Hook was not created."
    assert len(trial.steps()) > 0, "Nothing saved at any step."
    assert len(trial.tensor_names()) > 0, "Tensors were not saved."
