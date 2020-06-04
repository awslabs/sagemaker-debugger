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
    train_steps, eval_steps = 8, 2
    hook = smd.EstimatorHook(out_dir=out_dir, save_all=saveall)
    hook.set_mode(mode=smd.modes.TRAIN)
    mnist_classifier.train(input_fn=train_input_fn, steps=train_steps, hooks=[hook])
    hook.set_mode(mode=smd.modes.EVAL)
    mnist_classifier.evaluate(input_fn=eval_input_fn, steps=eval_steps, hooks=[hook])

    # Check that hook created and tensors saved
    trial = smd.create_trial(path=out_dir)
    tnames = trial.tensor_names()
    assert len(trial.steps()) > 0
    assert len(tnames) == 301 if saveall else len(tnames) == 1


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
    estimator.train(input_fn=train_input_fn, steps=10, hooks=[hook])

    # Check that hook created and tensors saved
    trial = smd.create_trial(path=out_dir)
    tnames = trial.tensor_names()
    assert len(trial.steps()) > 0
    assert len(tnames) == 224 if saveall else len(tnames) == 2
