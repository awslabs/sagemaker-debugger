"""
This script is a simple MNIST training script which uses Horovod and Tensorflow's Keras interface.
It is designed to be used with SageMaker Debugger in an official SageMaker Framework container (i.e. AWS Deep Learning Container).
You will notice that this script looks exactly like a normal TensorFlow training script.
The hook needed by SageMaker Debugger to save tensors during training will be automatically added in those environments.
The hook will load configuration from json configuration that SageMaker will put in the training container from the configuration provided using the SageMaker python SDK when creating a job.
For more information, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md

This script has been adapted from an example in Horovod repository https://github.com/uber/horovod
"""
# Standard Library
import argparse
import math
import os

# Third Party
import horovod.tensorflow.keras as hvd
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args):
    # Horovod: initialize Horovod.
    hvd.init()

    if not args.use_only_cpu:
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
    else:
        config = None

    K.set_session(tf.Session(config=config))

    batch_size = 128
    num_classes = 10

    # Horovod: adjust number of epochs based on number of GPUs.
    epochs = int(math.ceil(args.num_epochs / hvd.size()))

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    # Horovod: adjust learning rate based on number of GPUs.
    opt = keras.optimizers.Adadelta(1.0 * hvd.size())

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(os.path.join(args.model_dir, "checkpoint-{epoch}.h5"))
        )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1 if hvd.rank() == 0 else 0,
        validation_data=(x_test, y_test),
    )
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_only_cpu", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--model_dir", type=str, default="/tmp/mnist_model")
    args = parser.parse_args()

    main(args)
