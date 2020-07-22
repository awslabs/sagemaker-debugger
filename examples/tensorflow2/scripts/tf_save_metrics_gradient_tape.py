"""
This file is temporary, for testing with 2.X.
We'll need to integrate a more robust testing pipeline and make this part of pytest
before pushing to master.
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

    model = tf.keras.models.Sequential(
        [
            # WA for TF issue https://github.com/tensorflow/tensorflow/issues/36279
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
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
    hook.wrap_optimizer(opt)
    hook.register_model(model)  # Can be skipped in ZCC

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            labels = tf.one_hot(labels, depth=10)
            with hook.wrap_tape(tf.GradientTape(persistent=persistent)) as tape:
                logits = model(data, training=True)
                loss_value = cce(labels, logits)
            hook.save_tensor("y_labels", labels, "outputs")
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
            hook.record_tensor_value(tensor_name="accuracy", tensor_value=acc)
        train_acc_metric.reset_states()

    hook.close()


def test_keras_gradtape(out_dir):
    """
    Test save all and save default collection
    """
    include_collections = [
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.LAYERS,
        CollectionKeys.LOSSES,
        CollectionKeys.INPUTS,
        CollectionKeys.OUTPUTS,
        CollectionKeys.METRICS,
        CollectionKeys.OPTIMIZER_VARIABLES,
    ]
    hook = smd.KerasHook(
        out_dir=out_dir,
        save_config=SaveConfig(save_interval=1),
        include_collections=include_collections,
    )
    helper_keras_gradtape(trial_dir=out_dir, hook=hook)

    trial = smd.create_trial(path=out_dir)
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES)) == 5
    assert len(trial.tensor_names(collection=CollectionKeys.LAYERS)) == 8
    assert len(trial.tensor_names(collection=CollectionKeys.OUTPUTS)) == 2
    assert len(trial.tensor_names(collection=CollectionKeys.INPUTS)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) == 1
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) == 1

    # We assert that all the tensors saved have a valid value
    for tname in trial.tensor_names():
        assert trial.tensor(tname).value(5) is not None

    # We then analyse Layer Inputs and Layer Outputs
    # Check that output of a layer is equal to the input of the next
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
