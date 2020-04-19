# Future
from __future__ import absolute_import, division, print_function, unicode_literals

# Standard Library
import os

# Third Party
import pytest
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib
from tests.tensorflow2.utils import is_tf_2_2
from tests.tensorflow.utils import create_trial_fast_refresh

# First Party
import smdebug.tensorflow as smd
from smdebug.core.access_layer import has_training_ended
from smdebug.core.collection import CollectionKeys
from smdebug.core.modes import ModeKeys
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.exceptions import TensorUnavailable, TensorUnavailableForStep
from smdebug.tensorflow import ReductionConfig, SaveConfig
from smdebug.tensorflow.keras import KerasHook

tfds.disable_progress_bar()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == "GPU"])


def train_model(
    trial_dir,
    save_all=False,
    hook=None,
    include_collections=None,
    reduction_config=None,
    save_config=None,
    eager=True,
    strategy=None,
    steps=None,
    add_callbacks=None,
    include_workers="all",
):
    tf.keras.backend.clear_session()
    if not eager:
        tf.compat.v1.disable_eager_execution()

    datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)

    mnist_train, mnist_test = datasets["train"], datasets["test"]

    if strategy is None:
        strategy = tf.distribute.MirroredStrategy()

    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    if hook is None:
        if save_config is None:
            save_config = SaveConfig(save_interval=3)

        hook = KerasHook(
            out_dir=trial_dir,
            save_config=save_config,
            reduction_config=reduction_config,
            include_collections=include_collections,
            save_all=save_all,
            include_workers=include_workers,
        )

        if not save_all and include_collections is not None:
            for cname in hook.include_collections:
                if cname not in include_collections:
                    hook.get_collection(cname).save_config = SaveConfig(end_step=0)

    opt = tf.keras.optimizers.Adam()

    opt = hook.wrap_optimizer(opt)

    with strategy.scope():
        relu_layer = tf.keras.layers.Dense(64, activation="relu")
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                relu_layer,
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    hooks = []
    if add_callbacks:
        if "tensorboard" in add_callbacks:
            hooks.append(
                # write_grads = True causes crash saying handle must be created in scope
                # erorr like this https://stackoverflow.com/questions/56836895/custom-training-loop-using-tensorflow-gpu-1-14-and-tf-distribute-mirroredstrateg
                # this crash is even if callback is off
                tf.keras.callbacks.TensorBoard(
                    log_dir="/tmp/logs", histogram_freq=4, write_images=True
                )
            )

    hooks.append(hook)

    if steps is None:
        steps = ["train"]
    for step in steps:
        if step == "train":
            model.fit(train_dataset, epochs=1, steps_per_epoch=10, callbacks=hooks, verbose=0)
        elif step == "eval":
            model.evaluate(eval_dataset, steps=10, callbacks=hooks, verbose=0)
        elif step == "predict":
            model.predict(train_dataset, steps=4, callbacks=hooks, verbose=0)

    smd.get_hook().close()
    return strategy


def exhaustive_check(trial_dir, include_workers="one", eager=True):
    include_collections = [
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.LOSSES,
        CollectionKeys.OUTPUTS,
        CollectionKeys.METRICS,
        CollectionKeys.OPTIMIZER_VARIABLES,
    ]
    strategy = train_model(
        trial_dir,
        include_collections=include_collections,
        steps=["train", "eval", "predict", "train"],
        include_workers=include_workers,
        eager=eager,
    )

    tr = create_trial_fast_refresh(trial_dir)
    print(tr.tensor_names())

    if include_workers == "all":
        assert len(tr.workers()) == strategy.num_replicas_in_sync
        if eager:
            assert len(tr.tensor_names()) == (6 + 1 + 2 + 5 if is_tf_2_2() else 6 + 1 + 3 + 5)
            # 6 weights, 1 loss, 3 metrics, 5 optimizer variables for Tf 2.1
            # 6 weights, 1 loss, 2 metrics, 5 optimizer variables for Tf 2.2
        else:
            assert len(tr.tensor_names()) == (6 + 6 + 1 + 3 + strategy.num_replicas_in_sync * 3 + 5)
    else:
        assert len(tr.workers()) == 1
        if eager:
            assert len(tr.tensor_names()) == (6 + 3 + 1)
        else:
            assert len(tr.tensor_names()) == (6 + 6 + 1 + 3 + 1 * 3 + 5)

    # 6 weights, 6 gradients, 1 loss, 3 metrics, 24 outputs (8 for each mode), 5 optimizer variables
    assert len(tr.modes()) == 3
    assert len(tr.steps()) == 14
    assert len(tr.steps(ModeKeys.TRAIN)) == 8  # 0, 3, 6, 9, 12, 15, 18, 19(end of epoch)
    assert len(tr.steps(ModeKeys.EVAL)) == 4
    assert len(tr.steps(ModeKeys.PREDICT)) == 2  # ran 4 steps above

    assert len(tr.tensor_names(collection=CollectionKeys.BIASES)) == 3
    wtnames = tr.tensor_names(collection=CollectionKeys.WEIGHTS)
    assert len(wtnames) == 3

    for wtname in wtnames:
        assert len(tr.tensor(wtname).steps()) == 13, wtname
        assert len(tr.tensor(wtname).steps(ModeKeys.TRAIN)) == 7
        for s in tr.tensor(wtname).steps(ModeKeys.TRAIN):
            assert tr.tensor(wtname).value(s, mode=ModeKeys.TRAIN) is not None
            for worker in tr.workers():
                assert tr.tensor(wtname).value(s, mode=ModeKeys.TRAIN, worker=worker) is not None
        assert len(tr.tensor(wtname).steps(ModeKeys.EVAL)) == 4
        for s in tr.tensor(wtname).steps(ModeKeys.EVAL):
            assert tr.tensor(wtname).value(s, mode=ModeKeys.EVAL) is not None
            for worker in tr.workers():
                assert tr.tensor(wtname).value(s, mode=ModeKeys.EVAL, worker=worker) is not None
        assert len(tr.tensor(wtname).steps(ModeKeys.PREDICT)) == 2

    if not eager:
        gradnames = tr.tensor_names(collection=CollectionKeys.GRADIENTS)
        assert len(gradnames) == 6
        for gradname in gradnames:
            assert len(tr.tensor(gradname).steps(ModeKeys.TRAIN)) == 7
            for s in tr.tensor(gradname).steps(ModeKeys.TRAIN):
                assert tr.tensor(gradname).value(s, mode=ModeKeys.TRAIN) is not None
            assert len(tr.tensor(gradname).steps(ModeKeys.EVAL)) == 0
            assert len(tr.tensor(gradname).steps(ModeKeys.PREDICT)) == 0

        optvarnames = tr.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)
        assert len(optvarnames) == 5
        for optvarname in optvarnames:
            assert len(tr.tensor(optvarname).steps(ModeKeys.TRAIN)) == 7
            for s in tr.tensor(optvarname).steps(ModeKeys.TRAIN):
                assert tr.tensor(optvarname).value(s, mode=ModeKeys.TRAIN) is not None
            assert len(tr.tensor(optvarname).steps(ModeKeys.EVAL)) == 0
            assert len(tr.tensor(optvarname).steps(ModeKeys.PREDICT)) == 0

    assert len(tr.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    loss_name = tr.tensor_names(collection=CollectionKeys.LOSSES)[0]
    # loss is not in predict mode (so less 2)
    # add one for end of epoch
    assert len(tr.tensor(loss_name).steps(ModeKeys.TRAIN)) == 8
    assert len(tr.tensor(loss_name).steps(ModeKeys.EVAL)) == 4
    assert len(tr.tensor(loss_name).steps(ModeKeys.PREDICT)) == 0
    assert len(tr.tensor(loss_name).steps()) == 12

    metricnames = tr.tensor_names(collection=CollectionKeys.METRICS)
    assert len(metricnames) == (2 if is_tf_2_2() else 3)


@pytest.mark.slow
def test_tf_keras(out_dir, tf_eager_mode, include_workers="all"):
    exhaustive_check(out_dir, include_workers=include_workers, eager=tf_eager_mode)


@pytest.mark.slow
def test_save_all(out_dir, tf_eager_mode):
    strategy = train_model(
        out_dir,
        include_collections=None,
        save_all=True,
        save_config=SaveConfig(save_steps=[5]),
        steps=["train"],
        eager=tf_eager_mode,
    )
    tr = create_trial_fast_refresh(out_dir)
    print(tr.tensor_names())
    if tf_eager_mode:
        assert len(tr.tensor_names()) == (6 + 2 + 1 + 5 if is_tf_2_2() else 6 + 3 + 1 + 5)
        # weights, metrics, losses, optimizer variables
    else:
        assert (
            len(tr.tensor_names())
            == 6
            + 6
            + 5
            + 3
            + 1
            + 3 * strategy.num_replicas_in_sync
            + 2 * strategy.num_replicas_in_sync
        )
        # weights, grads, optimizer_variables, metrics, losses, outputs
    assert len(tr.steps()) == 3


@pytest.mark.slow
def test_save_one_worker(out_dir, tf_eager_mode):
    strategy = train_model(
        out_dir,
        include_collections=None,
        save_all=True,
        save_config=SaveConfig(save_steps=[5]),
        steps=["train"],
        include_workers="one",
        eager=tf_eager_mode,
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.workers()) == 1
    assert len(tr.steps())
    assert len(tr.tensor_names(collection="weights"))
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(5)) == 1
    assert len(tr.tensor_names(collection="biases"))
    assert len(tr.tensor(tr.tensor_names(collection="biases")[0]).workers(5)) == 1


@pytest.mark.slow
def test_save_all_workers(out_dir, tf_eager_mode):
    # Skip if no GPUS
    if get_available_gpus() == 0:
        return
    strategy = train_model(
        out_dir,
        include_collections=None,
        save_all=True,
        save_config=SaveConfig(save_steps=[5]),
        steps=["train"],
        include_workers="all",
        eager=tf_eager_mode,
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.workers()) == get_available_gpus()
    assert len(tr.tensor_names(collection="weights"))
    assert (
        len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(5))
        == strategy.num_replicas_in_sync
    )

    assert "conv2d/weights/conv2d/kernel:0" in tr.tensor_names(collection="weights")
    assert (
        len(tr.tensor("conv2d/weights/conv2d/kernel:0").workers(5)) == strategy.num_replicas_in_sync
    )

    assert len(tr.tensor_names(collection="biases"))
    assert "conv2d/weights/conv2d/bias:0" in tr.tensor_names(collection="biases")
    assert (
        len(tr.tensor(tr.tensor_names(collection="biases")[0]).workers(5))
        == strategy.num_replicas_in_sync
    )


@pytest.mark.slow
def test_base_reductions(out_dir, tf_eager_mode):
    train_model(
        out_dir,
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.METRICS,
            CollectionKeys.LOSSES,
        ],
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=ALLOWED_REDUCTIONS),
        steps=["train"],
        eager=tf_eager_mode,
    )

    tr = create_trial_fast_refresh(out_dir)
    weight_name = tr.tensor_names(collection=CollectionKeys.WEIGHTS)[0]

    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        assert tr.tensor(weight_name).reduction_values(0)

    loss_name = tr.tensor_names(collection=CollectionKeys.LOSSES)[0]
    assert tr.tensor(loss_name).value(0) is not None

    metric_name = tr.tensor_names(collection=CollectionKeys.METRICS)[0]
    assert tr.tensor(metric_name).value(0) is not None


@pytest.mark.slow
def test_collection_reductions(out_dir, tf_eager_mode):
    tf.keras.backend.clear_session()
    hook = KerasHook(
        out_dir=out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.GRADIENTS,
        ],
    )
    hook.get_collection(CollectionKeys.WEIGHTS).reduction_config = ReductionConfig(norms=["l1"])
    train_model(out_dir, hook=hook, steps=["train"], eager=tf_eager_mode)

    tr = create_trial_fast_refresh(out_dir)
    weight_name = tr.tensor_names(collection=CollectionKeys.WEIGHTS)[0]

    try:
        tr.tensor(weight_name).value(0)
        assert False
    except TensorUnavailableForStep:
        try:
            assert tr.tensor(weight_name).reduction_value(0, "l1") is not None
        except ValueError:
            # some tensors reduction can't be computed
            pass
    except TensorUnavailable:
        # sometimes we might not have tensor saved if it was only being
        # saved as reduction and the reduction computation failed
        pass


@pytest.mark.slow
def test_training_end(out_dir, tf_eager_mode):
    train_model(
        out_dir, include_collections=[CollectionKeys.OUTPUTS], steps=["train"], eager=tf_eager_mode
    )
    assert has_training_ended(out_dir) is True


@pytest.mark.slow
def test_include_regex(out_dir, tf_eager_mode):
    hook = KerasHook(
        out_dir=out_dir,
        save_config=SaveConfig(save_interval=9),
        include_collections=["custom_coll"],
        include_workers="all",
    )
    hook.get_collection("custom_coll").include("dense")
    strategy = train_model(out_dir, hook=hook, steps=["train"], eager=tf_eager_mode)

    tr = create_trial_fast_refresh(out_dir)
    tnames = tr.tensor_names(collection="custom_coll")

    if tf_eager_mode:
        assert len(tnames) == 4
    else:
        assert len(tnames) == 4 + 3 * strategy.num_replicas_in_sync
    for tname in tnames:
        assert tr.tensor(tname).value(0) is not None


@pytest.mark.skip_if_non_eager
@pytest.mark.slow
def test_clash_with_tb_callback(out_dir):
    train_model(
        out_dir,
        save_config=SaveConfig(save_interval=9),
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.GRADIENTS,
            CollectionKeys.LOSSES,
            CollectionKeys.METRICS,
        ],
        steps=["train"],
        add_callbacks=["tensorboard"],
    )
    tr = create_trial_fast_refresh(out_dir)
    assert len(tr.tensor_names()) == (9 if is_tf_2_2() else 10)


def test_one_device(out_dir, tf_eager_mode):
    strategy = train_model(
        out_dir,
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.OUTPUTS,
            CollectionKeys.GRADIENTS,
        ],
        save_config=SaveConfig(save_interval=9),
        strategy=tf.distribute.OneDeviceStrategy(device="/cpu:0"),
        steps=["train"],
        eager=tf_eager_mode,
    )
    assert os.path.isdir(os.path.join(out_dir, "events")) is False
