"""
This file is temporary, for testing with 2.X.
We'll need to integrate a more robust testing pipeline and make this part of pytest
before pushing to master.

This was tested with TensorFlow 2.1, by running
`python tests/tensorflow2/test_keras.py` from the main directory.
"""
# Standard Library
import json
import re
import time
from pathlib import Path

# Third Party
import pytest
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tests.tensorflow2.utils import is_greater_than_tf_2_2, is_tf_2_3, is_tf_2_6
from tests.tensorflow.utils import create_trial_fast_refresh
from tests.utils import verify_shapes

# First Party
import smdebug.tensorflow as smd
from smdebug.core.access_layer import has_training_ended
from smdebug.core.collection import CollectionKeys
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.modes import ModeKeys
from smdebug.core.reduction_config import ALLOWED_NORMS
from smdebug.exceptions import TensorUnavailableForStep
from smdebug.profiler.profiler_constants import DEFAULT_PREFIX
from smdebug.tensorflow import ReductionConfig, SaveConfig


def get_model():
    model = tf.keras.models.Sequential(
        [
            # WA for TF issue https://github.com/tensorflow/tensorflow/issues/36279
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


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
    run_eagerly=False,
):
    if not eager:
        tf.compat.v1.disable_eager_execution()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255, x_test / 255

    model = get_model()
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
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=run_eagerly,
    )
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

    model.save(trial_dir, save_format="tf")

    hook.close()


def helper_keras_gradtape(
    trial_dir,
    save_all=False,
    include_collections=None,
    reduction_config=None,
    save_config=None,
    hook=None,
    batch_size=64,
    persistent=False,
):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(batch_size)
    model = get_model()
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
    hook.wrap_optimizer(opt)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            labels = tf.one_hot(labels, depth=10)
            with hook.wrap_tape(tf.GradientTape(persistent=persistent)) as tape:
                logits = model(data, training=True)  # (32,10)
                loss_value = cce(labels, logits)
            grads = tape.gradient(loss_value, model.variables)

            # By default, the resources held by a GradientTape are released as
            # soon as GradientTape.gradient() method is called. To compute
            # multiple gradients over the same computation, create a persistent
            # gradient tape. This allows multiple calls to the gradient() method
            # as resources are released when the tape object is garbage collected.
            if persistent:
                _ = tape.gradient(loss_value, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            acc = train_acc_metric(dataset_labels, logits)
            hook.save_tensor(
                tensor_name="accuracy", tensor_value=acc, collections_to_write="metrics"
            )
        train_acc_metric.reset_states()

    model.save(trial_dir, save_format="tf")
    hook.close()


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_layer_names_gradient_tape(out_dir):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=9),
        include_collections=[CollectionKeys.LAYERS],
    )
    helper_keras_gradtape(out_dir, hook=hook, save_config=SaveConfig(save_interval=9))

    tr = create_trial_fast_refresh(out_dir)
    tnames = tr.tensor_names(collection=CollectionKeys.LAYERS)
    pattern = r"^(flatten|dense|dropout)(_\d+)?\/(inputs|outputs)"
    for tname in tnames:
        assert re.match(pattern=pattern, string=tname) is not None


@pytest.mark.skip_if_non_eager
def test_keras_gradtape_shapes(out_dir):
    hook = smd.KerasHook(
        out_dir=out_dir,
        save_all=True,
        save_config=SaveConfig(save_steps=[0]),
        reduction_config=ReductionConfig(save_shape=True),
    )
    helper_keras_gradtape(trial_dir=out_dir, hook=hook)
    verify_shapes(out_dir, 0)
    verify_shapes(out_dir, 500)


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
@pytest.mark.parametrize("saveall", [True, False])
def test_keras_gradtape(out_dir, saveall):
    """
    Test save all and save default collection
    """
    hook = smd.KerasHook(out_dir=out_dir, save_all=saveall, save_config=SaveConfig(save_interval=3))
    helper_keras_gradtape(trial_dir=out_dir, hook=hook)

    trial = smd.create_trial(path=out_dir)
    if saveall:  # save losses, metrics, weights, biases
        if is_tf_2_6():
            num_tensors = 15
        elif is_greater_than_tf_2_2():
            num_tensors = 25
        else:
            num_tensors = 15
        assert len(trial.tensor_names()) == num_tensors
        assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    else:  # save the default losses and metrics
        assert len(trial.tensor_names()) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_gradtape_base_reductions(out_dir):
    """
    Test reduction config
    """
    reductions = ["min", "max", "mean", "std", "sum", "prod"]
    helper_keras_gradtape(
        trial_dir=out_dir,
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.METRICS, CollectionKeys.LOSSES],
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=reductions),
    )
    tr = create_trial_fast_refresh(out_dir)
    weight_name = tr.tensor_names(collection=CollectionKeys.WEIGHTS)[0]
    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_value(0, "l1") is not None
        assert len(tr.tensor(weight_name).reduction_values(0)) == len(reductions) + len(
            ALLOWED_NORMS
        )

    loss_name = tr.tensor_names(collection=CollectionKeys.LOSSES)[0]
    assert tr.tensor(loss_name).value(0) is not None

    metric_name = tr.tensor_names(collection=CollectionKeys.METRICS)[0]
    assert tr.tensor(metric_name).value(0) is not None


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_gradtape_collection_reductions(out_dir):
    """
    Test reduction config for weights collection
    """
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.BIASES],
    )
    hook.get_collection(CollectionKeys.WEIGHTS).reduction_config = ReductionConfig(norms=["l1"])
    helper_keras_gradtape(out_dir, hook=hook)

    tr = create_trial_fast_refresh(out_dir)
    weight_name = tr.tensor_names(collection=CollectionKeys.WEIGHTS)[0]
    bias_name = tr.tensor_names(collection=CollectionKeys.BIASES)[0]

    assert tr.tensor(bias_name).value(0) is not None
    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_value(0, "l1") is not None


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_gradtape_include_regex(out_dir):
    """
    Test custom collection with regex
    """
    hook = smd.KerasHook(
        out_dir, save_config=SaveConfig(save_interval=9), include_collections=["custom_coll"]
    )
    hook.get_collection("custom_coll").include("dense")
    helper_keras_gradtape(out_dir, hook=hook, save_config=SaveConfig(save_interval=9))

    tr = create_trial_fast_refresh(out_dir)
    tnames = tr.tensor_names(collection="custom_coll")
    if is_tf_2_6():
        num_tensors = 8
    elif is_greater_than_tf_2_2():
        num_tensors = 12
    else:
        num_tensors = 8
    assert len(tnames) == num_tensors
    for tname in tnames:
        assert tr.tensor(tname).value(0) is not None


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_gradtape_training_end(out_dir):
    """
    Verify enf of training file
    """
    helper_keras_gradtape(out_dir, include_collections=[CollectionKeys.OUTPUTS])
    assert has_training_ended(out_dir) is True


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_gradtape_weights_collections(out_dir):
    """
    This test ensures that a training script written with GradientTape
    handles the case where only weight and default collections are saved.
    The aim is to distinguish between weights and biases.
    """
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.WEIGHTS],
    )

    helper_keras_gradtape(out_dir, hook=hook)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_gradtape_include_collections(out_dir):
    """
    This test ensures that a training script written with GradientTape
    handles the case where hook config contains all collections mentioned
    through include collections
    """
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
        reduction_config=ReductionConfig(
            norms=ALLOWED_NORMS, reductions=["min", "max", "mean", "std", "sum", "prod"]
        ),
    )
    helper_keras_gradtape(out_dir, hook=hook)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    if is_tf_2_6():
        num_tensors = 15
    elif is_greater_than_tf_2_2():
        num_tensors = 16
    else:
        num_tensors = 15
    assert len(trial.tensor_names()) == num_tensors
    assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_gradtape_hook_from_json(out_dir, monkeypatch):
    """
    This test ensures that a training script written with GradientTape
    handles the case where hook config is provided through a JSON and saves
    only weights and losses
    """
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_collection_defaults.json",
    )
    hook = smd.KerasHook.create_from_json_file()
    helper_keras_gradtape(out_dir, hook=hook)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
@pytest.mark.parametrize("saveall", [True, False])
def test_gradtape_persistent(out_dir, saveall):
    """
    Test save all and save default collection
    """
    hook = smd.KerasHook(out_dir=out_dir, save_all=saveall, save_config=SaveConfig(save_interval=3))
    helper_keras_gradtape(trial_dir=out_dir, hook=hook, persistent=True)

    trial = smd.create_trial(path=out_dir)
    if saveall:  # save losses, metrics, weights, biases
        if is_tf_2_6():
            num_tensors = 15
        elif is_greater_than_tf_2_2():
            num_tensors = 25
        else:
            num_tensors = 15
        assert len(trial.tensor_names()) == num_tensors
        assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    else:  # save the default losses and metrics
        assert len(trial.tensor_names()) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


@pytest.mark.slow
@pytest.mark.parametrize("saveall", [True, False])
def test_keras_fit(out_dir, tf_eager_mode, saveall):
    hook = smd.KerasHook(out_dir=out_dir, save_all=saveall)
    ts = time.time()
    hook.save_scalar("foobar", 1, sm_metric=True, timestamp=ts)
    scalars_to_be_saved = dict()
    scalars_to_be_saved["scalar/foobar"] = (ts, 0)
    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook,
        run_eagerly=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x eager mode
    if saveall:  # save losses, metrics, weights, biases, scalar
        if tf_eager_mode:
            if is_tf_2_6():
                num_tensors = 13
            elif is_greater_than_tf_2_2():
                num_tensors = 28
            elif is_tf_2_3():
                num_tensors = 21
            else:
                num_tensors = 14
            assert len(trial.tensor_names()) == num_tensors
            assert len(trial.tensor_names(collection=CollectionKeys.INPUTS)) == (
                1 if is_greater_than_tf_2_2() and not is_tf_2_6() else 0
            )
            assert len(trial.tensor_names(collection=CollectionKeys.OUTPUTS)) == (
                2 if is_greater_than_tf_2_2() and not is_tf_2_6() else 0
            )
        else:
            assert len(trial.tensor_names()) == 21
        assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
        assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
        assert (
            len(
                trial.tensor_names(
                    collection=CollectionKeys.OPTIMIZER_VARIABLES, mode=ModeKeys.EVAL
                )
            )
            == 0,
            "No Optimizer Variables Should be Saved in EVAL Mode",
        )
    else:  # save the default losses and metrics
        assert len(trial.tensor_names()) == (
            4 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 5
        )
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == (
        2 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 3
    )
    for tname in trial.tensor_names():
        assert trial.tensor(tname).value(0) is not None


def test_keras_fit_shapes(out_dir):
    hook = smd.KerasHook(
        out_dir=out_dir,
        save_all=True,
        save_config=SaveConfig(save_steps=[0]),
        reduction_config=ReductionConfig(save_shape=True),
    )
    helper_keras_fit(trial_dir=out_dir, hook=hook)
    print(create_trial_fast_refresh(out_dir).tensor_names(step=0))
    verify_shapes(out_dir, 0)


@pytest.mark.slow
def test_base_reductions(out_dir, tf_eager_mode):
    reductions = ["min", "max", "mean", "std", "sum", "prod"]
    helper_keras_fit(
        trial_dir=out_dir,
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.METRICS, CollectionKeys.LOSSES],
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=reductions),
        run_eagerly=tf_eager_mode,
    )
    tr = create_trial_fast_refresh(out_dir)
    weight_name = tr.tensor_names(collection=CollectionKeys.WEIGHTS)[0]
    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_value(0, "l1") is not None
        assert len(tr.tensor(weight_name).reduction_values(0)) == len(reductions) + len(
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
        run_eagerly=tf_eager_mode,
    )

    tr = create_trial_fast_refresh(out_dir)
    tnames = tr.tensor_names(collection="custom_coll")
    assert len(tnames) == (12 if is_greater_than_tf_2_2() and not is_tf_2_6() else 4)
    for tname in tnames:
        assert tr.tensor(tname).value(0) is not None


@pytest.mark.slow
def test_layer_names(out_dir, tf_eager_mode):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=9),
        include_collections=[CollectionKeys.LAYERS],
    )
    helper_keras_fit(
        out_dir,
        hook=hook,
        save_config=SaveConfig(save_interval=9),
        steps=["train"],
        run_eagerly=tf_eager_mode,
    )

    tr = create_trial_fast_refresh(out_dir)
    tnames = tr.tensor_names(collection=CollectionKeys.LAYERS)
    pattern = r"^(flatten|dense|dropout)(_\d+)?\/(inputs|outputs)"
    for tname in tnames:
        assert re.match(pattern=pattern, string=tname) is not None


@pytest.mark.slow
@pytest.mark.skipif(is_tf_2_6(), reason="Unsupported with TF 2.6 due to keras breaking changes")
def test_regex_filtering_for_default_collections(out_dir):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=9),
        include_collections=[CollectionKeys.LAYERS, CollectionKeys.GRADIENTS],
    )
    hook.get_collection(CollectionKeys.LAYERS).include("^dense")
    hook.get_collection(CollectionKeys.GRADIENTS).include("gradients/dense")
    helper_keras_fit(
        out_dir,
        hook=hook,
        save_config=SaveConfig(save_interval=10),
        steps=["train"],
        run_eagerly=True,
    )

    tr = create_trial_fast_refresh(out_dir)
    layer_tnames = tr.tensor_names(collection=CollectionKeys.LAYERS)
    gradient_tnames = tr.tensor_names(collection=CollectionKeys.GRADIENTS)
    assert len(layer_tnames) == (4 if is_greater_than_tf_2_2() else 0)
    assert len(gradient_tnames) == (4 if is_greater_than_tf_2_2() else 0)
    layer_pattern = r"^(dense)(_\d+)?\/(inputs|outputs)"
    gradient_pattern = r"gradients/dense"
    for tname in layer_tnames:
        assert tr.tensor(tname).value(0) is not None
        assert re.match(pattern=layer_pattern, string=tname) is not None
    for tname in gradient_tnames:
        assert tr.tensor(tname).value(0) is not None
        assert re.match(pattern=gradient_pattern, string=tname) is not None


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_clash_with_tb_callback(out_dir):
    # this test cannot be run in non-eager mode
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
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensor_names()) == (7 if (is_greater_than_tf_2_2() or is_tf_2_3()) else 8)


@pytest.mark.slow
def test_training_end(out_dir, tf_eager_mode):
    helper_keras_fit(
        out_dir,
        include_collections=[CollectionKeys.OUTPUTS],
        steps=["train"],
        run_eagerly=tf_eager_mode,
    )
    assert has_training_ended(out_dir) is True


@pytest.mark.slow
def test_weights_collections(out_dir, tf_eager_mode):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.WEIGHTS],
    )

    helper_keras_fit(out_dir, hook=hook, steps=["train"], run_eagerly=tf_eager_mode)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == (
        5 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 6
    )
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == (
        2 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 3
    )


@pytest.mark.slow
def test_include_collections(out_dir, tf_eager_mode):
    include_collections = [
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.LOSSES,
        CollectionKeys.METRICS,
        CollectionKeys.OPTIMIZER_VARIABLES,
        "custom_optimizer_variables",
    ]
    save_config = SaveConfig(save_interval=3)
    hook = smd.KerasHook(
        out_dir,
        save_config=save_config,
        include_collections=include_collections,
        reduction_config=ReductionConfig(
            norms=ALLOWED_NORMS, reductions=["min", "max", "mean", "std", "sum", "prod"]
        ),
    )
    hook.get_collection("custom_optimizer_variables").include("Adam")
    helper_keras_fit(
        out_dir, hook=hook, steps=["train", "eval", "predict"], run_eagerly=tf_eager_mode
    )

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    if tf_eager_mode:
        if is_tf_2_6():
            assert len(trial.tensor_names()) == 12
        elif is_greater_than_tf_2_2():
            assert len(trial.tensor_names()) == 16
        else:
            assert len(trial.tensor_names()) == (12 if is_tf_2_3() else 13)
    else:
        assert len(trial.tensor_names()) == 18
        assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    assert len(trial.tensor_names(collection="custom_optimizer_variables")) == 5
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == (
        2 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 3
    )


@pytest.mark.slow
def test_include_only_custom_collection(out_dir, tf_eager_mode):
    include_collections = ["custom_optimizer_variables"]
    save_config = SaveConfig(save_interval=3)
    hook = smd.KerasHook(
        out_dir,
        save_config=save_config,
        include_collections=include_collections,
        reduction_config=ReductionConfig(
            norms=ALLOWED_NORMS, reductions=["min", "max", "mean", "std", "sum", "prod"]
        ),
    )
    hook.get_collection("custom_optimizer_variables").include("Adam")
    helper_keras_fit(
        out_dir, hook=hook, steps=["train", "eval", "predict"], run_eagerly=tf_eager_mode
    )

    trial = smd.create_trial(path=out_dir)
    assert len(trial.tensor_names()) == (
        8 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 9
    )
    assert len(trial.tensor_names(collection="custom_optimizer_variables")) == 5


@pytest.mark.slow
def test_hook_from_json(out_dir, tf_eager_mode, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_collection_defaults.json",
    )
    hook = smd.KerasHook.create_from_json_file()
    helper_keras_fit(out_dir, hook=hook, steps=["train"], run_eagerly=tf_eager_mode)

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == (
        5 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 6
    )
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == (
        2 if (is_greater_than_tf_2_2() or is_tf_2_3()) and tf_eager_mode else 3
    )


@pytest.mark.skip_if_non_eager
def test_keras_fit_pure_eager(out_dir, tf_eager_mode):
    """
    Test save all and save default collection in fit() pure eager mode
    """
    hook = smd.KerasHook(out_dir=out_dir, save_all=True, save_config=SaveConfig(save_interval=3))
    helper_keras_fit(trial_dir=out_dir, hook=hook, eager=tf_eager_mode, run_eagerly=True)

    trial = smd.create_trial(path=out_dir)
    if is_tf_2_6():
        assert len(trial.tensor_names()) == 12
    elif is_greater_than_tf_2_2():
        assert len(trial.tensor_names()) == 27
    else:
        assert len(trial.tensor_names()) == 13
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    assert len(trial.tensor_names(collection=CollectionKeys.INPUTS)) == (
        1 if is_greater_than_tf_2_2() and not is_tf_2_6() else 0
    )
    assert len(trial.tensor_names(collection=CollectionKeys.OUTPUTS)) == (
        2 if is_greater_than_tf_2_2() and not is_tf_2_6() else 0
    )


@pytest.mark.slow
@pytest.mark.skipif(is_tf_2_6(), reason="Unsupported with TF 2.6 due to keras breaking changes")
def test_model_inputs_and_outputs(out_dir, tf_eager_mode):
    # explicitly save INPUTS and OUTPUTS
    include_collections = [CollectionKeys.INPUTS, CollectionKeys.OUTPUTS]
    hook = smd.KerasHook(out_dir=out_dir, include_collections=include_collections)

    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook,
        eager=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )
    trial = smd.create_trial(path=out_dir)
    assert len(trial.steps(mode=ModeKeys.TRAIN)) == 3
    assert len(trial.tensor_names(collection=CollectionKeys.OUTPUTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.INPUTS)) == 1

    # Check the shape of output tensors
    assert trial.labels(step=0)[0].shape == (6000, 1)
    assert trial.predictions(step=0)[0].shape == (6000, 10)
    # Check the shape of input tensors
    assert trial.inputs(step=0)[0].shape == (6000, 28, 28)


@pytest.mark.skip  # skip until aws tf update
def test_save_gradients(out_dir, tf_eager_mode):
    # explicitly save INPUTS and OUTPUTS
    include_collections = [CollectionKeys.GRADIENTS]
    hook = smd.KerasHook(out_dir=out_dir, include_collections=include_collections)

    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook,
        eager=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )
    trial = smd.create_trial(path=out_dir)
    assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) == 4

    for tname in trial.tensor_names(collection=CollectionKeys.GRADIENTS):
        output = trial.tensor(tname)
        assert output.value(0) is not None


def test_save_tensors(out_dir, tf_eager_mode):
    include_collections = ["custom_coll"]
    hook = smd.KerasHook(out_dir=out_dir, include_collections=include_collections)
    t1 = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    t2 = tf.Variable([5 + 4j, 6 + 1j])
    t3 = tf.Variable([False, False, False, True])
    hook.save_tensor("custom_tensor_1", t1, include_collections)
    hook.save_tensor("custom_tensor_2", t2, include_collections)
    hook.save_tensor("custom_tensor_3", t3, include_collections)

    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook,
        eager=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )
    trial = smd.create_trial(path=out_dir)
    assert len(trial.steps(mode=ModeKeys.TRAIN)) == 3
    for tname in trial.tensor_names(collection="custom_coll"):
        assert trial.tensor(tname).value(0) is not None


def test_keras_to_estimator(out_dir, tf_eager_mode):
    if not tf_eager_mode:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
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

    keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=out_dir)

    hook = smd.EstimatorHook(out_dir)

    hook.set_mode(smd.modes.TRAIN)
    keras_estimator.train(input_fn=input_fn, steps=25, hooks=[hook])

    hook.set_mode(smd.modes.EVAL)
    eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10, hooks=[hook])

    from smdebug.trials import create_trial

    tr = create_trial(out_dir)
    assert len(tr.tensor_names()) == 1
    assert len(tr.steps()) == 2
    assert len(tr.steps(smd.modes.TRAIN)) == 1
    assert len(tr.steps(smd.modes.EVAL)) == 1


@pytest.mark.skip
def test_save_layer_inputs_and_outputs(out_dir, tf_eager_mode):
    # explicitly save INPUTS and OUTPUTS
    include_collections = [CollectionKeys.INPUTS, CollectionKeys.OUTPUTS]
    hook = smd.KerasHook(out_dir=out_dir, include_collections=include_collections)

    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook,
        eager=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )
    trial = smd.create_trial(path=out_dir)
    assert len(trial.tensor_names(collection=CollectionKeys.INPUTS)) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.OUTPUTS)) == 4

    # Check that output of layer is equal to the input of the next
    boolean_matrix = trial.tensor("flatten/outputs").value(0) == trial.tensor("dense/inputs").value(
        0
    )
    assert boolean_matrix.all()
    boolean_matrix = trial.tensor("dense/outputs").value(0) == trial.tensor("dropout/inputs").value(
        0
    )
    assert boolean_matrix.all()
    boolean_matrix = trial.tensor("dropout/outputs").value(0) == trial.tensor(
        "dense_1/inputs"
    ).value(0)
    assert boolean_matrix.all()


def test_hook_timeline_file_write(
    set_up_smprofiler_config_path, set_up_resource_config, out_dir, tf_eager_mode
):
    hook = smd.KerasHook(out_dir=out_dir, save_all=False)
    helper_keras_fit(trial_dir=out_dir, hook=hook, eager=tf_eager_mode, steps=["train", "eval"])

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict
