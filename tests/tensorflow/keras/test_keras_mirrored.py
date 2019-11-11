from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pytest
import shutil
import tensorflow_datasets as tfds
from datetime import datetime

import tensorflow as tf

tfds.disable_progress_bar()

from tornasole.exceptions import TensorUnavailableForStep, TensorUnavailable
from tornasole.core.modes import ModeKeys
from tornasole.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from tornasole.tensorflow import reset_collections, get_collection, SaveConfig, ReductionConfig
from tornasole.tensorflow.keras import TornasoleKerasHook
from tests.tensorflow.utils import create_trial_fast_refresh
from tornasole.core.collection import CollectionKeys
from tornasole.core.access_layer import has_training_ended
from tensorflow.python.keras.distribute.distributed_training_utils import get_distributed_model
from tensorflow.python.keras.utils.mode_keys import ModeKeys as KerasModeKeys

TORNASOLE_TF_HOOK_TESTS_DIR = "/tmp/tornasole_tf/tests/"


class FetchTensorCallback(tf.keras.callbacks.Callback):
    def __init__(self, tensors):
        self.tensors = tensors

    def _callback_fn(self, tensor_val):
        assert tensor_val is not None

    def on_train_batch_begin(self, batch, logs):
        for t in self.tensors:
            x = get_distributed_model(self.model, KerasModeKeys.TRAIN)._distributed_function
            x.fetches.append(t)
            x.fetch_callbacks[t] = self._callback_fn

    def on_train_batch_end(self, batch, logs):
        for t in self.tensors:
            x = get_distributed_model(self.model, KerasModeKeys.TRAIN)._distributed_function
            x.fetches.remove(t)
            del x.fetch_callbacks[t]


def train_model(
    trial_dir,
    save_all=False,
    include_collections=None,
    reduction_config=None,
    save_config=None,
    use_keras_optimizer=True,
    reset=True,
    create_relu_collection=False,
    steps=None,
    add_callbacks=None,
):
    print(tf.__version__)
    if reset:
        reset_collections()
        # tf.reset_default_graph()

    datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)

    mnist_train, mnist_test = datasets["train"], datasets["test"]

    strategy = tf.distribute.MirroredStrategy()

    # You can also do info.splits.total_num_examples to get the total
    # number of examples in the dataset.

    num_train_examples = info.splits["train"].num_examples
    num_test_examples = info.splits["test"].num_examples

    BUFFER_SIZE = 10000

    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

    if save_config is None:
        save_config = SaveConfig(save_interval=3)

    # include_collections = [
    #     # CollectionKeys.WEIGHTS,
    #     # CollectionKeys.GRADIENTS,
    #     # CollectionKeys.OPTIMIZER_VARIABLES,
    #     CollectionKeys.DEFAULT,
    #     # CollectionKeys.METRICS,
    #     # CollectionKeys.LOSSES,
    #     # CollectionKeys.OUTPUTS,
    #     # CollectionKeys.SCALARS,
    # ]

    hook = TornasoleKerasHook(
        out_dir=trial_dir,
        save_config=save_config,
        reduction_config=reduction_config,
        include_collections=include_collections,
        save_all=save_all,
    )

    if use_keras_optimizer:
        opt = tf.keras.optimizers.Adam()
    else:
        opt = tf.train.AdamOptimizer(0.1)

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

    if create_relu_collection:
        get_collection("relu").add_keras_layer(relu_layer, inputs=True, outputs=True)

    # get_collection('default').include('Relu')

    hooks = []
    if add_callbacks:
        if "tensorboard" in add_callbacks:
            hooks.append(
                # write_grads = True causes crash saying handle must be created in scope
                # erorr like this https://stackoverflow.com/questions/56836895/custom-training-loop-using-tensorflow-gpu-1-14-and-tf-distribute-mirroredstrateg
                # this crash is even if tornasole callback is off
                tf.keras.callbacks.TensorBoard(
                    log_dir="./logs", histogram_freq=4, write_images=True
                )
            )
        if "fetch_tensor" in add_callbacks:
            hooks.append(FetchTensorCallback(model.weights))
    hooks.append(hook)

    # model.fit(train_dataset, epochs=1, callbacks=callbacks)
    # model.predict(eval_dataset, callbacks=callbacks)
    # model.fit(train_dataset, epochs=1, callbacks=callbacks)

    if steps is None:
        steps = ["train"]
    for step in steps:
        if step == "train":
            model.fit(train_dataset, epochs=1, steps_per_epoch=10, callbacks=hooks, verbose=0)
        elif step == "eval":
            model.evaluate(eval_dataset, steps=10, callbacks=hooks, verbose=0)
        elif step == "predict":
            model.predict(train_dataset, steps=4, callbacks=hooks, verbose=0)

    hook._cleanup()
    return strategy
    # model.fit(x_train, y_train, epochs=1, steps_per_epoch=10, callbacks=hooks, verbose=0)
    # model.evaluate(x_test, y_test, steps=10, callbacks=hooks, verbose=0)
    # model.predict(x_test[:100], callbacks=hooks, verbose=0)


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


def exhaustive_check():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    include_collections = [
        CollectionKeys.WEIGHTS,
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
    )

    tr = create_trial_fast_refresh(trial_dir)
    print(tr.tensors())
    assert len(tr.tensors()) == (6 + 6 + 1 + 3 + strategy.num_replicas_in_sync * 3 + 5)
    # 6 weights, 6 gradients, 1 loss, 3 metrics, 24 outputs (8 for each mode), 5 optimizer variables
    assert len(tr.modes()) == 3
    assert len(tr.steps()) == 14
    assert len(tr.steps(ModeKeys.TRAIN)) == 8  # 0, 3, 6, 9, 12, 15, 18, 19(end of epoch)
    assert len(tr.steps(ModeKeys.EVAL)) == 4
    assert len(tr.steps(ModeKeys.PREDICT)) == 2  # ran 4 steps above

    wtnames = tr.tensors_in_collection(CollectionKeys.WEIGHTS)
    assert len(wtnames) == 6
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

    gradnames = tr.tensors_in_collection(CollectionKeys.GRADIENTS)
    assert len(gradnames) == 6
    for gradname in gradnames:
        assert len(tr.tensor(gradname).steps(ModeKeys.TRAIN)) == 7
        for s in tr.tensor(gradname).steps(ModeKeys.TRAIN):
            assert tr.tensor(gradname).value(s, mode=ModeKeys.TRAIN) is not None
        assert len(tr.tensor(gradname).steps(ModeKeys.EVAL)) == 0
        assert len(tr.tensor(gradname).steps(ModeKeys.PREDICT)) == 0

    optvarnames = tr.tensors_in_collection(CollectionKeys.OPTIMIZER_VARIABLES)
    assert len(optvarnames) == 5
    for optvarname in optvarnames:
        assert len(tr.tensor(optvarname).steps(ModeKeys.TRAIN)) == 7
        for s in tr.tensor(optvarname).steps(ModeKeys.TRAIN):
            assert tr.tensor(optvarname).value(s, mode=ModeKeys.TRAIN) is not None
        assert len(tr.tensor(optvarname).steps(ModeKeys.EVAL)) == 0
        assert len(tr.tensor(optvarname).steps(ModeKeys.PREDICT)) == 0

    assert len(tr.tensors_in_collection(CollectionKeys.LOSSES)) == 1
    loss_name = tr.tensors_in_collection(CollectionKeys.LOSSES)[0]
    # loss is not in predict mode (so less 2)
    # add one for end of epoch
    assert len(tr.tensor(loss_name).steps(ModeKeys.TRAIN)) == 8
    assert len(tr.tensor(loss_name).steps(ModeKeys.EVAL)) == 4
    assert len(tr.tensor(loss_name).steps(ModeKeys.PREDICT)) == 0
    assert len(tr.tensor(loss_name).steps()) == 12

    metricnames = tr.tensors_in_collection(CollectionKeys.METRICS)
    assert len(metricnames) == 3
    shutil.rmtree(trial_dir)


@pytest.mark.slow
def test_tf_keras():
    exhaustive_check()


@pytest.mark.slow
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
        use_keras_optimizer=False,
        steps=["train", "eval"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    assert len(tr.modes()) == 2
    assert len(tr.steps(ModeKeys.TRAIN)) == 4  # 0, 3, 6, 9
    assert len(tr.tensors_in_collection(CollectionKeys.GRADIENTS)) == 6
    gradient_name = tr.tensors_in_collection(CollectionKeys.GRADIENTS)[0]
    assert len(tr.tensor(gradient_name).steps(ModeKeys.TRAIN)) == 4
    assert len(tr.tensor(gradient_name).steps(ModeKeys.EVAL)) == 0

    # not supported for non keras optimizer with keras
    assert len(tr.tensors_in_collection(CollectionKeys.OPTIMIZER_VARIABLES)) == 0
    shutil.rmtree(trial_dir)


@pytest.mark.slow
def test_save_all():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    strategy = train_model(
        trial_dir,
        include_collections=None,
        save_all=True,
        save_config=SaveConfig(save_steps=[5]),
        steps=["train"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    print(tr.tensors())
    assert (
        len(tr.tensors())
        == 6 + 6 + 5 + 3 + 1 + 3 * strategy.num_replicas_in_sync + 2 * strategy.num_replicas_in_sync
    )
    # weights, grads, optimizer_variables, metrics, losses, outputs
    assert len(tr.steps()) == 3
    shutil.rmtree(trial_dir)


@pytest.mark.slow
@pytest.mark.skip("https://github.com/awslabs/tornasole_core/issues/377")
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
    for tname in tr.tensors():
        for s in tr.tensor(tname).steps():
            print(tname, tr.tensor(tname).reduction_values(0))

    f = open(
        os.path.join(
            trial_dir, "index", "000000000", "000000000000__replica-0_task-0_device-CPU-0.json"
        )
    )
    print(f.readlines())
    f.close()

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


@pytest.mark.slow
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
        try:
            assert tr.tensor(weight_name).reduction_value(0, "l1") is not None
        except ValueError:
            # some tensors reduction can't be computed
            pass
    except TensorUnavailable:
        # sometimes we might not have tensor saved if it was only being
        # saved as reduction and the reduction computation failed
        pass
    shutil.rmtree(trial_dir)


@pytest.mark.slow
def test_training_end():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    train_model(trial_dir, include_collections=[CollectionKeys.OUTPUTS], steps=["train"])
    assert has_training_ended(trial_dir) is True
    shutil.rmtree(trial_dir)


@pytest.mark.slow
def test_collection_add():
    reset_collections()
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    strategy = train_model(
        trial_dir,
        include_collections=["relu"],
        reset=False,
        save_config=SaveConfig(save_interval=9),
        create_relu_collection=True,
        steps=["train"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    relu_coll_tensor_names = tr.tensors_in_collection("relu")
    assert len(relu_coll_tensor_names) == strategy.num_replicas_in_sync * 2
    assert tr.tensor(relu_coll_tensor_names[0]).value(0) is not None
    assert tr.tensor(relu_coll_tensor_names[1]).value(0) is not None
    shutil.rmtree(trial_dir)


@pytest.mark.slow
def test_include_regex():
    reset_collections()
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    get_collection("custom_coll").include("dense")
    strategy = train_model(
        trial_dir,
        include_collections=["custom_coll"],
        save_config=SaveConfig(save_interval=9),
        reset=False,
        steps=["train"],
    )
    tr = create_trial_fast_refresh(trial_dir)
    print(tr.tensors())
    tnames = tr.tensors_in_collection("custom_coll")
    assert len(tnames) == 4 + 3 * strategy.num_replicas_in_sync
    for tname in tnames:
        assert tr.tensor(tname).value(0) is not None
    shutil.rmtree(trial_dir)


@pytest.mark.slow
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
    assert len(tr.tensors()) == 16
    shutil.rmtree(trial_dir)


@pytest.mark.slow
def test_clash_with_custom_callback():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    strategy = train_model(
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
    print(tr.tensors())
    assert len(tr.tensors()) == 6 + 6 + strategy.num_replicas_in_sync * 1 + 3
    shutil.rmtree(trial_dir)
