# Standard Library
# Third Party
import pytest
import tensorflow.compat.v2 as tf
from tests.tensorflow2.utils import is_tf_version_greater_than_2_4_x
from tests.zero_code_change.tf_utils import get_estimator, get_input_fns

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys


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
    if saveall:
        # Number of tensors in each collection
        # vanilla TF 2.2: all = 300, loss = 1, weights = 4, gradients = 0, biases = 18, optimizer variables = 0, metrics = 0, others = 277
        # AWS-TF 2.2 : all = 300, loss = 1, weights = 4, gradients = 8, biases = 18, optimizer variables = 0, metrics = 0, others = 269
        # AWS-TF 2.1 : all = 309, loss = 1, weights = 4, gradients = 8, biases = 18, optimizer variables = 0, metrics = 0, others = 278
        assert len(tnames) >= 1 + 4 + 18
        assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
        assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 4
        assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 18
        assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) >= 0
        assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) >= 0
    else:
        assert len(tnames) == 1
        assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1


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
    if saveall:
        # Number of tensors in each collection
        # vanilla TF 2.2: all = 214, loss = 2, weights = 1, gradients = 0, biases = 12, optimizer variables = 0, metrics = 0, others = 199
        # AWS-TF 2.2: all = 219, loss = 2, weights = 1, gradients = 2, biases = 12, optimizer variables = 5, metrics = 0, others = 197
        # AWS-TF 2.1: all = 226, loss = 2, weights = 1, gradients = 2, biases = 12, optimizer variables = 5, metrics = 0, others = 204
        # AWS-TF 2.4: all = 229, loss = 2, weights = 1, gradients = 2, biases = 16, optimizer variables = 5, metrics = 0, others = 197
        assert len(tnames) >= 2 + 1 + 12
        assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 1
        assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == (
            16 if is_tf_version_greater_than_2_4_x() else 12
        )
        assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) >= 0
        assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) >= 0
    else:
        assert len(tnames) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 2
