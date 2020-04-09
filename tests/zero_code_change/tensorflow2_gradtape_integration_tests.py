"""
WARNING: This test is useful for DLC testing.

Test with Keras GradientTape.

We check that certain tensors are saved.
Here in the test suite we delete the hook after every script.
"""
# Standard Library
import argparse

# Third Party
import tensorflow.compat.v2 as tf

# First Party
import smdebug.tensorflow as smd
from smdebug.core.utils import SagemakerSimulator


def get_keras_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (x_train, y_train), (x_test, y_test)


def helper_test_keras_v2_gradienttape(script_mode: bool = False, json_file_contents="{}"):
    """ Test the default ZCC behavior of saving losses and metrics in eager and non-eager modes."""
    smd.del_hook()
    tf.keras.backend.clear_session()

    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # WA for TF issue #36279
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        (x_train, y_train), _ = get_keras_data()
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
        )
        dataset = dataset.shuffle(1000).batch(64)

        opt = tf.keras.optimizers.RMSprop()
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        n_epochs = 2
        if script_mode:
            if json_file_contents == "{}":
                hook = smd.KerasHook(out_dir=sim.out_dir, export_tensorboard=True)
            else:
                hook = smd.KerasHook.create_from_json_file()

            for epoch in range(n_epochs):
                print("Epoch %d/%d" % (epoch + 1, n_epochs))
                for data, labels in dataset:
                    dataset_labels = labels
                    labels = tf.one_hot(labels, depth=10)
                    with hook.wrap_tape(tf.GradientTape()) as tape:
                        logits = model(data, training=True)  # (32,10)
                        loss_value = cce(labels, logits)
                    grads = tape.gradient(loss_value, model.variables)
                    opt.apply_gradients(zip(grads, model.variables))
                    acc = train_acc_metric(dataset_labels, logits)
                    hook.record_tensor_value(tensor_name="accuracy", tensor_value=acc)
                log = "Epoch %d " % (epoch + 1)
                log += "Accuracy %.4f" % train_acc_metric.result()
                print(log)
                train_acc_metric.reset_states()
            hook = smd.get_hook()
            assert hook
            hook.close()
            # Check that hook created and tensors saved
            trial = smd.create_trial(path=sim.out_dir)
            assert len(trial.steps()) > 0, "Nothing saved at any step."
            assert len(trial.tensor_names()) > 0, "Tensors were not saved."
            assert len(trial.tensor_names(collection="losses")) > 0
        else:
            # ZCC doesn't support yet (as of smdebug v0.7.2)
            for epoch in range(n_epochs):
                print("Epoch %d/%d" % (epoch + 1, n_epochs))
                for data, labels in dataset:
                    dataset_labels = labels
                    labels = tf.one_hot(labels, depth=10)
                    with tf.GradientTape(persistent=True) as tape:
                        logits = model(data, training=True)  # (32,10)
                        loss_value = cce(labels, logits)
                    grads = tape.gradient(loss_value, model.variables)
                    opt.apply_gradients(zip(grads, model.variables))
                    acc = train_acc_metric(dataset_labels, logits)
                log = "Epoch %d " % (epoch + 1)
                log += "Accuracy %.4f" % train_acc_metric.result()
                print(log)
                train_acc_metric.reset_states()
            hook = smd.get_hook()
            assert not hook


def test_keras_v2_default(script_mode: bool = False):
    # Test default ZCC behavior
    helper_test_keras_v2_gradienttape(script_mode=script_mode)


def test_keras_v2_multi_collections(script_mode: bool = False):
    # Test multiple collections included in hook json
    json_file_contents = """
            {
                "S3OutputPath": "s3://sagemaker-test",
                "LocalPath": "/opt/ml/output/tensors",
                "HookParameters" : {
                    "save_interval": "100",
                    "include_workers": "all"
                },
                "CollectionConfigurations": [
                    {
                        "CollectionName": "gradients"
                    },
                    {
                        "CollectionName": "weights"
                    },
                    {
                        "CollectionName": "losses"
                    },
                    {
                        "CollectionName": "biases"
                    },
                    {
                        "CollectionName": "optimizer_variables"
                    }
                ]
            }
            """
    helper_test_keras_v2_gradienttape(
        script_mode=script_mode, json_file_contents=json_file_contents
    )


def test_keras_v2_save_all(script_mode: bool = False):
    # Test save all through hook config
    json_file_contents = """
            {
                "S3OutputPath": "s3://sagemaker-test",
                "LocalPath": "/opt/ml/output/tensors",
                "HookParameters" : {
                    "save_steps": "0,1,2,3",
                    "save_all": true
                }
            }
            """
    helper_test_keras_v2_gradienttape(
        script_mode=script_mode, json_file_contents=json_file_contents
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-mode", help="Manually create hooks instead of relying on ZCC", action="store_true"
    )
    args = parser.parse_args()
    script_mode = args.script_mode

    # Gradient Tape eager mode
    test_keras_v2_default(script_mode)
    test_keras_v2_multi_collections(script_mode)
    test_keras_v2_save_all(script_mode)
