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


def helper_keras_gradtape(
    trial_dir,
    save_all=False,
    include_collections=None,
    reduction_config=None,
    save_config=None,
    hook=None,
    steps=None,
    batch_size=32,
    add_callbacks=None,
):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(batch_size)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
    #                            input_shape=(None, None, 1)),
    #     tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(10)
    # ])
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

    # tape = tf.GradientTape(persistent=True)
    # tape = hook.wrap_tape(tape)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # hooks = []
    # hooks.append(hook)

    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            labels = tf.one_hot(labels, depth=10)
            with tf.GradientTape(persistent=True) as tape:
                logits = model(data, training=True)  # (32,10)
                loss_value = cce(labels, logits)
                # layer = model.layers[1]
                # vars = layer
            grads = tape.gradient(loss_value, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            acc = train_acc_metric(dataset_labels, logits)
            hook.update_step(grads, model.variables, loss_value, acc, tape)
        train_acc_metric.reset_states()

    hook.close()


@pytest.mark.slow
@pytest.mark.parametrize("saveall", [True, False])
def test_keras_gradtape(out_dir, saveall):
    hook = smd.KerasHook(out_dir=out_dir, save_all=saveall)
    helper_keras_gradtape(trial_dir=out_dir, hook=hook, steps=["train", "eval", "predict", "train"])

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x eager mode
    if saveall:  # save losses, metrics, weights, biases
        assert len(trial.tensor_names()) == 22
        # assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
        # assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    else:  # save the default losses and metrics
        assert len(trial.tensor_names()) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


def test_base_reductions(out_dir):
    helper_keras_gradtape(
        trial_dir=out_dir,
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.METRICS, CollectionKeys.LOSSES],
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=ALLOWED_REDUCTIONS),
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
def test_collection_reductions(out_dir):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.BIASES],
    )
    hook.get_collection(CollectionKeys.WEIGHTS).reduction_config = ReductionConfig(norms=["l1"])
    helper_keras_gradtape(out_dir, hook=hook, steps=["train"])

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
def test_include_regex(out_dir):
    hook = smd.KerasHook(
        out_dir, save_config=SaveConfig(save_interval=9), include_collections=["custom_coll"]
    )
    hook.get_collection("custom_coll").include("dense")
    helper_keras_gradtape(
        out_dir, hook=hook, save_config=SaveConfig(save_interval=9), steps=["train"]
    )

    tr = create_trial_fast_refresh(out_dir)
    tnames = tr.tensor_names(collection="custom_coll")

    assert len(tnames) == 4
    for tname in tnames:
        assert tr.tensor(tname).value(0) is not None


@pytest.mark.slow
def test_training_end(out_dir):
    helper_keras_gradtape(out_dir, include_collections=[CollectionKeys.OUTPUTS], steps=["train"])
    assert has_training_ended(out_dir) is True


@pytest.mark.slow
def test_weights_collections(out_dir):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.WEIGHTS],
    )

    helper_keras_gradtape(out_dir, hook=hook, steps=["train"])

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


@pytest.mark.slow
def test_include_collections(out_dir):
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
    helper_keras_gradtape(out_dir, hook=hook, steps=["train", "eval", "predict"])

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == 18
    assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1


@pytest.mark.slow
def test_hook_from_json(out_dir, monkeypatch):
    monkeypatch.setenv(
        CONFIG_FILE_PATH_ENV_STR,
        "tests/tensorflow/hooks/test_json_configs/test_collection_defaults.json",
    )
    hook = smd.KerasHook.create_from_json_file()
    helper_keras_gradtape(out_dir, hook=hook, steps=["train"])

    trial = smd.create_trial(path=out_dir)
    # can't save gradients in TF 2.x
    assert len(trial.tensor_names()) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1
