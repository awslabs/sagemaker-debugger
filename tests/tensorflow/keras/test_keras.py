# Standard Library
import os
import shutil
from datetime import datetime

# Third Party
import pytest
import tensorflow as tf
from tests.tensorflow.utils import create_trial_fast_refresh

# First Party
from smdebug.core.access_layer import has_training_ended
from smdebug.core.collection import CollectionKeys
from smdebug.core.modes import ModeKeys
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.exceptions import TensorUnavailableForStep
from smdebug.tensorflow import ReductionConfig, SaveConfig, get_collection, reset_collections
from smdebug.tensorflow.keras import TornasoleKerasHook

TORNASOLE_TF_HOOK_TESTS_DIR = "/tmp/tornasole_tf/tests/"


class FetchTensorCallback(tf.keras.callbacks.Callback):
    def __init__(self, tensors):
        self.tensors = tensors

    def _callback_fn(self, tensor_val):
        assert tensor_val is not None

    def on_train_batch_begin(self, batch, logs):
        for t in self.tensors:
            self.model.train_function.fetches.append(t)
            self.model.train_function.fetch_callbacks[t] = self._callback_fn

    def on_train_batch_end(self, batch, logs):
        for t in self.tensors:
            self.model.train_function.fetches.remove(t)
            del self.model.train_function.fetch_callbacks[t]


def train_model(
    trial_dir,
    save_all=False,
    include_collections=None,
    reduction_config=None,
    save_config=None,
    use_tf_keras=True,
    eager=False,
    use_keras_optimizer=True,
    reset=True,
    create_relu_collection=False,
    steps=None,
    add_callbacks=None,
):
    if use_tf_keras:
        from tensorflow import keras
    else:
        import keras

    if reset:
        reset_collections()
        tf.reset_default_graph()

    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    relu_layer = keras.layers.Dense(128, activation="relu")
    if create_relu_collection:
        get_collection("relu").add_keras_layer(relu_layer, inputs=True, outputs=True)

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            relu_layer,
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    if save_config is None:
        save_config = SaveConfig(save_interval=3)

    hook = TornasoleKerasHook(
        trial_dir,
        save_config=save_config,
        save_all=save_all,
        include_collections=include_collections,
        reduction_config=reduction_config,
    )

    if use_keras_optimizer:
        opt = keras.optimizers.RMSprop()
    else:
        opt = tf.train.RMSPropOptimizer(0.1)

    opt = hook.wrap_optimizer(opt)

    if use_tf_keras:
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            run_eagerly=eager,
            metrics=["accuracy"],
        )
    else:
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    hooks = []
    if add_callbacks:
        if "tensorboard" in add_callbacks:
            hooks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir="./logs", histogram_freq=1, write_grads=True, write_images=True
                )
            )
        if "fetch_tensor" in add_callbacks:
            hooks.append(FetchTensorCallback(model.outputs + model.weights))
    hooks.append(hook)

    if steps is None:
        steps = ["train"]
    for step in steps:
        if step == "train":
            model.fit(x_train, y_train, epochs=1, steps_per_epoch=10, callbacks=hooks, verbose=0)
        elif step == "eval":
            model.evaluate(x_test, y_test, steps=10, callbacks=hooks, verbose=0)
        elif step == "predict":
            model.predict(x_test[:100], callbacks=hooks, verbose=0)

    hook._cleanup()


@pytest.mark.skip(
    "needs to be run individually as it complains that eager "
    "needs to be set at startup, but pytest "
    "does not allow controlling order of tests"
)
def test_tf_keras_eager():
    tf.enable_eager_execution()
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(trial_dir, eager=True, steps=["train"])
    tf.disable_eager_execution()
    shutil.rmtree(trial_dir, ignore_errors=True)


@pytest.mark.skip(
    "needs to be run individually as it complains that eager "
    "needs to be set at startup, but pytest "
    "does not allow controlling order of tests"
)
def test_tf_keras_eager_env():
    tf.enable_eager_execution()
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(trial_dir, eager=False, steps=["train"])
    tf.disable_eager_execution()
    shutil.rmtree(trial_dir, ignore_errors=True)


def exhaustive_check(use_tf_keras):
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    include_collections = [
        CollectionKeys.WEIGHTS,
        CollectionKeys.GRADIENTS,
        CollectionKeys.LOSSES,
        CollectionKeys.OUTPUTS,
        CollectionKeys.METRICS,
        CollectionKeys.LOSSES,
        CollectionKeys.OPTIMIZER_VARIABLES,
    ]
    train_model(
        trial_dir,
        include_collections=include_collections,
        use_tf_keras=use_tf_keras,
        eager=False,
        steps=["train", "eval", "predict", "train"],
    )

    tr = create_trial_fast_refresh(trial_dir)
    if use_tf_keras:
        assert len(tr.tensors()) == 18
    else:
        # can't save optimizer variables in this case
        assert len(tr.tensors()) == 13

    assert len(tr.modes()) == 3
    assert len(tr.steps(ModeKeys.TRAIN)) == 8  # 0, 3, 6, 9, 12, 15, 18, 19(end of epoch)
    assert len(tr.steps(ModeKeys.EVAL)) == 4
    assert len(tr.steps(ModeKeys.PREDICT)) == 2  # ran 4 steps above

    assert len(tr.tensors_in_collection(CollectionKeys.GRADIENTS)) == 4
    gradient_name = tr.tensors_in_collection(CollectionKeys.GRADIENTS)[0]
    assert len(tr.tensor(gradient_name).steps(ModeKeys.TRAIN)) == 7
    assert len(tr.tensor(gradient_name).steps(ModeKeys.EVAL)) == 0

    assert len(tr.tensors_in_collection(CollectionKeys.WEIGHTS)) == 4
    weight_name = tr.tensors_in_collection(CollectionKeys.WEIGHTS)[0]
    assert len(tr.tensor(weight_name).steps()) == 13
    assert len(tr.tensor(weight_name).steps(ModeKeys.TRAIN)) == 7
    assert len(tr.tensor(weight_name).steps(ModeKeys.EVAL)) == 4

    assert len(tr.tensors_in_collection(CollectionKeys.LOSSES)) == 1
    loss_name = tr.tensors_in_collection(CollectionKeys.LOSSES)[0]
    assert len(tr.tensor(loss_name).steps()) == 12

    assert len(tr.tensors_in_collection(CollectionKeys.METRICS)) == 3

    if use_tf_keras:
        assert len(tr.tensors_in_collection(CollectionKeys.OPTIMIZER_VARIABLES)) == 5
        opt_var_name = tr.tensors_in_collection(CollectionKeys.OPTIMIZER_VARIABLES)[0]
        assert tr.tensor(opt_var_name).value(0) is not None
        assert len(tr.tensor(opt_var_name).steps(ModeKeys.EVAL)) == 0

    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:08 to run
def test_keras():
    exhaustive_check(False)


@pytest.mark.slow  # 0:07 to run
def test_tf_keras():
    exhaustive_check(True)


@pytest.mark.slow  # 0:03 to run
def test_tf_keras_non_keras_opt():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    include_collections = [
        CollectionKeys.GRADIENTS,
        CollectionKeys.OPTIMIZER_VARIABLES,
        CollectionKeys.METRICS,
    ]
    train_model(
        trial_dir,
        include_collections=include_collections,
        eager=False,
        use_keras_optimizer=False,
        steps=["train", "eval"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    assert len(tr.modes()) == 2
    assert len(tr.steps(ModeKeys.TRAIN)) == 4  # 0, 3, 6, 9
    assert len(tr.tensors_in_collection(CollectionKeys.GRADIENTS)) == 4
    gradient_name = tr.tensors_in_collection(CollectionKeys.GRADIENTS)[0]
    assert len(tr.tensor(gradient_name).steps(ModeKeys.TRAIN)) == 4
    assert len(tr.tensor(gradient_name).steps(ModeKeys.EVAL)) == 0

    # not supported for non keras optimizer with keras
    assert len(tr.tensors_in_collection(CollectionKeys.OPTIMIZER_VARIABLES)) == 0
    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:09 to run
def test_save_all():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(trial_dir, include_collections=None, save_all=True, steps=["train"])
    tr = create_trial_fast_refresh(trial_dir)
    assert len(tr.tensors()) == 21
    assert len(tr.steps()) == 4
    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:03 to run
def test_base_reductions():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(
        trial_dir,
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.METRICS, CollectionKeys.LOSSES],
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=ALLOWED_REDUCTIONS),
        steps=["train"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    print(tr.tensors())
    weight_name = tr.tensors_in_collection(CollectionKeys.WEIGHTS)[0]
    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_value(0, "l1") is not None
        assert len(tr.tensor(weight_name).reduction_values(0)) == len(ALLOWED_REDUCTIONS) + len(
            ALLOWED_NORMS
        )

    loss_name = tr.tensors_in_collection(CollectionKeys.LOSSES)[0]
    assert tr.tensor(loss_name).value(0) is not None

    metric_name = tr.tensors_in_collection(CollectionKeys.METRICS)[0]
    assert tr.tensor(metric_name).value(0) is not None

    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:03 to run
def test_collection_reductions():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)

    reset_collections()
    tf.reset_default_graph()

    get_collection(CollectionKeys.GRADIENTS).reduction_config = ReductionConfig(norms=["l1"])
    train_model(
        trial_dir,
        reset=False,
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.GRADIENTS],
        steps=["train"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    weight_name = tr.tensors_in_collection(CollectionKeys.WEIGHTS)[0]
    grad_name = tr.tensors_in_collection(CollectionKeys.GRADIENTS)[0]
    assert tr.tensor(weight_name).value(0) is not None
    try:
        tr.tensor(grad_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_value(0, "l1") is not None

    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:03 to run
def test_training_end():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(trial_dir, include_collections=[CollectionKeys.OUTPUTS], steps=["train"])

    assert has_training_ended(trial_dir) is True
    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:06 to run
def test_collection_add():
    reset_collections()
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(
        trial_dir,
        include_collections=["relu"],
        reset=False,
        save_config=SaveConfig(save_interval=9),
        create_relu_collection=True,
        steps=["train"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    relu_coll_tensor_names = tr.tensors_in_collection("relu")
    assert len(relu_coll_tensor_names) == 2
    assert tr.tensor(relu_coll_tensor_names[0]).value(0) is not None
    assert tr.tensor(relu_coll_tensor_names[1]).value(0) is not None
    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:06 to run
def test_include_regex():
    reset_collections()
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    get_collection("custom_coll").include("dense")
    train_model(
        trial_dir,
        include_collections=["custom_coll"],
        save_config=SaveConfig(save_interval=9),
        reset=False,
        steps=["train"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    tnames = tr.tensors_in_collection("custom_coll")
    assert len(tnames) == 8
    for tname in tnames:
        assert tr.tensor(tname).value(0) is not None
    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:03 to run
def test_clash_with_tb_callback():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(
        trial_dir,
        save_config=SaveConfig(save_interval=9),
        steps=["train"],
        add_callbacks=["tensorboard"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    assert len(tr.tensors()) == 12
    shutil.rmtree(trial_dir)


@pytest.mark.slow  # 0:03 to run
def test_clash_with_custom_callback():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(
        trial_dir,
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.OUTPUTS,
            CollectionKeys.GRADIENTS,
        ],
        save_config=SaveConfig(save_interval=9),
        steps=["train"],
        add_callbacks=["fetch_tensor"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    assert len(tr.tensors()) == 11
    shutil.rmtree(trial_dir)
