"""
This script is a ResNet training script which uses Tensorflow's Keras interface.
It has been orchestrated with SageMaker Debugger hook to allow saving tensors during training.
Here, the hook has been created using its constructor to allow running this locally for your experimentation.
When you want to run this script in SageMaker, it is recommended to create the hook from json file.
Please see scripts in either 'sagemaker_byoc' or 'sagemaker_official_container' folder based on your use case.
"""

# Standard Library
import argparse

# Third Party
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# First Party
import smdebug.tensorflow as smd


def train(batch_size, epoch, model, hook):
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()

    Y_train = to_categorical(y_train, 10)
    Y_valid = to_categorical(y_valid, 10)

    X_train = X_train.astype("float32")
    X_valid = X_valid.astype("float32")

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_valid -= mean_image
    X_train /= 128.0
    X_valid /= 128.0

    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_valid, Y_valid),
        shuffle=True,
        ##### Enabling SageMaker Debugger ###########
        # adding hook as a callback
        callbacks=[hook],
    )


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
        include_collections=["weights", "gradients", "losses"],
        save_config=smd.SaveConfig(save_interval=opt.save_interval),
    )

    optimizer = tf.keras.optimizers.Adam()

    ##### Enabling SageMaker Debugger ###########
    # wrap the optimizer so the hook can identify the gradients
    optimizer = hook.wrap_optimizer(optimizer)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # start the training.
    train(opt.batch_size, opt.epoch, model, hook)


if __name__ == "__main__":
    main()
