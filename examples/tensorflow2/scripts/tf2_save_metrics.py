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

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.tensorflow import SaveConfig


@pytest.fixture(scope="function")
def out_dir():
    """ Use this method to construct an out_dir.

    Then it will be automatically cleaned up for you, passed into the test method, and we'll have
    fewer folders lying around.
    """
    out_dir = "/tmp/test"
    shutil.rmtree(out_dir, ignore_errors=True)
    return out_dir


def helper_keras_fit(
    trial_dir,
    save_all=False,
    include_collections=None,
    reduction_config=None,
    save_config=None,
    hook=None,
    steps=None,
    add_callbacks=None,
    run_eagerly=False,
):

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(TEST_DATASET_S3_PATH)
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

    hook.close()


def test_keras_fit_eager(out_dir, tf_eager_mode=True):
    test_include_collections = [
        CollectionKeys.LOSSES,
        CollectionKeys.METRICS,
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.INPUTS,
        CollectionKeys.OUTPUTS,
        CollectionKeys.LAYERS,
        CollectionKeys.OPTIMIZER_VARIABLES,
    ]
    hook = smd.KerasHook(out_dir=out_dir, include_collections=test_include_collections)
    helper_keras_fit(
        include_collections=test_include_collections,
        trial_dir=out_dir,
        hook=hook,
        run_eagerly=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )
    trial = smd.create_trial(path=out_dir)

    # We first assert that none of the collections we requested for are empty
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.INPUTS)) == 1  # 1 Model Input
    assert len(trial.tensor_names(collection=CollectionKeys.OUTPUTS)) == 2  # 2 Model outputs
    assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5

    # We assert that all the tensors saved have a valid value
    for tname in trial.tensor_names():
        assert trial.tensor(tname).value(0) is not None

    # We then analyse Layer Inputs and Layer Outputs
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


def test_keras_fit_false(out_dir, tf_eager_mode=False):
    test_include_collections = [
        CollectionKeys.LOSSES,
        CollectionKeys.METRICS,
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.INPUTS,
        CollectionKeys.OUTPUTS,
        CollectionKeys.LAYERS,
        CollectionKeys.OPTIMIZER_VARIABLES,
    ]
    hook = smd.KerasHook(out_dir=out_dir, include_collections=test_include_collections)
    helper_keras_fit(
        include_collections=test_include_collections,
        trial_dir=out_dir,
        hook=hook,
        run_eagerly=tf_eager_mode,
        steps=["train", "eval", "predict", "train"],
    )
    trial = smd.create_trial(path=out_dir)

    # We first assert that none of the collections we requested for are empty
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.GRADIENTS)) == 4
    assert len(trial.tensor_names(collection=CollectionKeys.INPUTS)) == 1  # 1 Model Input
    assert len(trial.tensor_names(collection=CollectionKeys.OUTPUTS)) == 2  # 2 Model outputs
    assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5

    # We assert that all the tensors saved have a valid value
    for tname in trial.tensor_names():
        assert trial.tensor(tname).value(0) is not None

    # We then analyse Layer Inputs and Layer Outputs
    # Check that output of layer is equal to the input of the next
    boolean_matrix = trial.tensor("flatten_1/outputs").value(0) == trial.tensor(
        "dense_2/inputs"
    ).value(0)
    assert boolean_matrix.all()
    boolean_matrix = trial.tensor("dense_2/outputs").value(0) == trial.tensor(
        "dropout_1/inputs"
    ).value(0)
    assert boolean_matrix.all()
    boolean_matrix = trial.tensor("dropout_1/outputs").value(0) == trial.tensor(
        "dense_3/inputs"
    ).value(0)
    assert boolean_matrix.all()
