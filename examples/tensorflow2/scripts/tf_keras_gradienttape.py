"""
This script is a ResNet training script which uses Tensorflow's Keras interface.
It has been orchestrated with SageMaker Debugger hook to allow saving tensors during training.
Here, the hook has been created using its constructor to allow running this locally for your experimentation.
When you want to run this script in SageMaker, it is recommended to create the hook from json file.
"""

# Standard Library
import argparse

# Third Party
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# First Party
import smdebug.tensorflow as smd


def train(batch_size, n_epochs, model, hook):
    (x_train, y_train), _ = cifar10.load_data()

    x_train = x_train.astype("float32")

    mean_image = np.mean(x_train, axis=(0, 1, 2))
    stddev_image = np.std(x_train, axis=(0, 1, 2))
    x_train -= mean_image
    x_train /= stddev_image

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam()
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    num_training_samples = x_train.shape[0]

    # save_scalar() API can be used to save arbitrary scalar values that may
    # or may not be related to training.
    # Ref: https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#common-hook-api
    hook.save_scalar("num_training_samples", num_training_samples, sm_metric=True)

    for epoch in range(n_epochs):
        print("Epoch %d/%d" % (epoch + 1, n_epochs))
        progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics="accuracy")
        for idx, (data, labels) in enumerate(dataset):
            dataset_labels = labels
            labels = to_categorical(labels, 10)
            hook.save_scalar("step_number", idx + 1, sm_metric=False)
            # wrap the tape so the hook can identify the gradients, parameters, loss
            # wrapping the tape ensures that smdebug's wrapper around functions of
            # the tape object - push_tape(), pop_tape(), gradient(), will setup the writers of
            # the debugger and save tensors that are provided as input to gradient() (trainable
            # variables and loss), output of gradient() (gradients).
            with hook.wrap_tape(tf.GradientTape()) as tape:
                logits = model(data, training=True)  # (32,10)
                loss_value = cce(labels, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            acc = train_acc_metric(dataset_labels, logits)
            # save metrics value
            hook.record_tensor_value(tensor_name="accuracy", tensor_value=acc)
            values = [("Accuracy", train_acc_metric.result())]
            progBar.update(idx * batch_size, values=values)

        train_acc_metric.reset_states()
    hook.save_scalar("n_epochs", n_epochs, sm_metric=True)


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--save_interval", type=int, default=500)
    opt = parser.parse_args()

    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

    ##### Enabling SageMaker Debugger ###########
    # creating hook
    hook = smd.KerasHook(
        out_dir=opt.out_dir,
        # Information on default collections https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#default-collections-saved
        include_collections=["weights", "biases", "default", "gradients"],
        save_config=smd.SaveConfig(save_interval=opt.save_interval),
    )

    # start the training.
    train(opt.batch_size, opt.epoch, model, hook)


if __name__ == "__main__":
    main()
