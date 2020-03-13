"""
This file is temporary, for testing with 2.X.
We'll need to integrate a more robust testing pipeline and make this part of pytest
before pushing to master.

This was tested with TensorFlow 2.1, by running
`python tests/tensorflow2/test_keras.py` from the main directory.
"""

# Standard Library
import shutil

# Third Party
import pytest
import tensorflow.compat.v2 as tf
from tests.tensorflow.utils import create_trial_fast_refresh

# First Party
import smdebug.tensorflow as smd
from smdebug.core.access_layer import has_training_ended
from smdebug.core.collection import CollectionKeys
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.exceptions import TensorUnavailableForStep
from smdebug.tensorflow import ReductionConfig, SaveConfig


def helper_keras_fit(
    trial_dir,
    save_all=False,
    include_collections=None,
    reduction_config=None,
    save_config=None,
    hook=None,
    eager=True,
    steps=None,
    add_callbacks=None,
):
    if not eager:
        tf.compat.v1.disable_eager_execution()

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

    if hook is None:
        if save_config is None:
            save_config = SaveConfig(save_interval=3)

        hook = smd.KerasHook(
            trial_dir,
            save_config=save_config,
            save_all=save_all,
            include_collections=include_collections,
            reduction_config=reduction_config,
        )

        if not save_all and include_collections is not None:
            for cname in hook.include_collections:
                if cname not in include_collections:
                    hook.get_collection(cname).save_config = SaveConfig(end_step=0)

    opt = tf.keras.optimizers.Adam()

    opt = hook.wrap_optimizer(opt)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hooks = []
    if add_callbacks:
        if "tensorboard" in add_callbacks:
            hooks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir="/tmp/logs", histogram_freq=1, write_grads=True, write_images=True
                )
            )
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

    hook.close()


@pytest.mark.slow
@pytest.mark.parametrize("saveall", [True, False])
def test_keras_fit(out_dir, tf_eager_mode, saveall):
    hook = smd.KerasHook(out_dir=out_dir, save_all=saveall)
    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook,
        eager=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x eager mode
    if saveall:  # save losses, metrics, weights, biases
        if tf_eager_mode:
            assert len(trial.tensor_names()) == 8
        else:
            assert len(trial.tensor_names()) == 21
        assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    else:  # save the default losses and metrics
        assert len(trial.tensor_names()) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 3


@pytest.mark.slow
def test_base_reductions(out_dir, tf_eager_mode):
    helper_keras_fit(
        trial_dir=out_dir,
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.METRICS, CollectionKeys.LOSSES],
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=ALLOWED_REDUCTIONS),
        eager=tf_eager_mode,
    )
    tr = create_trial_fast_refresh(out_dir)
    weight_name = tr.tensor_names(collection=CollectionKeys.WEIGHTS)[0]
    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_value(0, "l1") is not None
        assert len(tr.tensor(weight_name).reduction_values(0)) == len(ALLOWED_REDUCTIONS) + len(
            ALLOWED_NORMS
        )

    loss_name = tr.tensor_names(collection=CollectionKeys.LOSSES)[0]
    assert tr.tensor(loss_name).value(0) is not None

    metric_name = tr.tensor_names(collection=CollectionKeys.METRICS)[0]
    assert tr.tensor(metric_name).value(0) is not None


@pytest.mark.slow
def test_collection_reductions(out_dir, tf_eager_mode):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.BIASES],
    )
    hook.get_collection(CollectionKeys.WEIGHTS).reduction_config = ReductionConfig(norms=["l1"])
    helper_keras_fit(out_dir, hook=hook, steps=["train"], eager=tf_eager_mode)

    tr = create_trial_fast_refresh(out_dir)
    weight_name = tr.tensor_names(collection=CollectionKeys.WEIGHTS)[0]
    bias_name = tr.tensor_names(collection=CollectionKeys.BIASES)[0]

    assert tr.tensor(bias_name).value(0) is not None
    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_value(0, "l1") is not None


@pytest.mark.slow
def test_include_regex(out_dir, tf_eager_mode):
    hook = smd.KerasHook(
        out_dir, save_config=SaveConfig(save_interval=9), include_collections=["custom_coll"]
    )
    hook.get_collection("custom_coll").include("dense")
    helper_keras_fit(
        out_dir,
        hook=hook,
        save_config=SaveConfig(save_interval=9),
        steps=["train"],
        eager=tf_eager_mode,
    )

    tr = create_trial_fast_refresh(out_dir)
    tnames = tr.tensor_names(collection="custom_coll")

    if tf_eager_mode:
        assert len(tnames) == 4
    else:
        assert len(tnames) == 8
    for tname in tnames:
        assert tr.tensor(tname).value(0) is not None


@pytest.mark.slow
def test_clash_with_tb_callback(out_dir, tf_eager_mode):
    # this test cannot be run in non-eager mode
    if not tf_eager_mode:
        return
    helper_keras_fit(
        out_dir,
        save_config=SaveConfig(save_interval=9),
        steps=["train"],
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.LOSSES,
            CollectionKeys.METRICS,
        ],
        add_callbacks=["tensorboard"],
        eager=tf_eager_mode,
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensor_names()) == 8


@pytest.mark.slow
def test_training_end(out_dir, tf_eager_mode):
    helper_keras_fit(
        out_dir, include_collections=[CollectionKeys.OUTPUTS], steps=["train"], eager=tf_eager_mode
    )
    assert has_training_ended(out_dir) is True


@pytest.mark.slow
def test_weights_collections(out_dir, tf_eager_mode):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.WEIGHTS],
    )

    helper_keras_fit(out_dir, hook=hook, steps=["train"], eager=tf_eager_mode)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == 6
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 3


@pytest.mark.slow
def test_include_collections(out_dir, tf_eager_mode):
    include_collections = [
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.LOSSES,
        CollectionKeys.OUTPUTS,
        CollectionKeys.METRICS,
        CollectionKeys.OPTIMIZER_VARIABLES,
    ]
    save_config = SaveConfig(save_interval=3)
    hook = smd.KerasHook(
        out_dir,
        save_config=save_config,
        include_collections=include_collections,
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=ALLOWED_REDUCTIONS),
    )
    helper_keras_fit(out_dir, hook=hook, steps=["train", "eval", "predict"], eager=tf_eager_mode)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    if tf_eager_mode:
        assert len(trial.tensor_names()) == 8
    else:
        assert len(trial.tensor_names()) == 18
        assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) == 4
        assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 3


@pytest.mark.slow
def test_hook_from_json(tf_eager_mode, monkeypatch):
    out_dir = "/tmp/test"
    shutil.rmtree(out_dir, ignore_errors=True)
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_collection_defaults.json",
    )
    hook = smd.KerasHook.create_from_json_file()
    helper_keras_fit(out_dir, hook=hook, steps=["train"], eager=tf_eager_mode)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == 6
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 3
