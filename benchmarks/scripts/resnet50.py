# Standard Library
import argparse
import json
import os
import random
import time

# Third Party
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


class TimeDuration(Callback):
    def on_train_begin(self, logs={}):
        self.train_start_time = time.time()
        print(f"Train_Start_Time={self.train_start_time};")

    def on_train_end(self, logs={}):
        self.train_duration = time.time() - self.train_start_time
        print(f"Train_End_Time={self.train_start_time + self.train_duration};")


def train(batch_size, epoch, model):
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

    time_callback = TimeDuration()
    callbacks = [time_callback]

    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_valid, Y_valid),
        shuffle=True,
        callbacks=callbacks,
    )
    print(f"Total_Train_Duration={time_callback.train_duration};")


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument("--write_profiler_config", type=bool, default=False)
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Number of steps to train for. If this" "is passed, it overrides num_epochs",
    )
    parser.add_argument(
        "--num_eval_steps",
        type=int,
        help="Number of steps to evaluate for. If this"
        "is passed, it doesnt evaluate over the full eval set",
    )
    args = parser.parse_args()

    if args.random_seed:
        tf.random.set_seed(2)
        np.random.seed(2)
        random.seed(12)

    if args.write_profiler_config:
        profiler_dict = {}
        profiler_dict["ProfilingIntervalInMilliseconds"] = 500
        param_dict = {}
        param_dict["ProfilerEnabled"] = "True"
        param_dict["LocalPath"] = "/opt/ml/output/tensors/"
        profiler_dict["ProfilingParameters"] = param_dict
        with open("/home/profilerconfig.json", "w") as f:
            json.dump(profiler_dict, f)
        os.environ["SMPROFILER_CONFIG_PATH"] = "/home/profilerconfig.json"

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
        opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # start the training.
    train(args.batch_size, args.epoch, model)


if __name__ == "__main__":
    os.system("mkdir -p ~/.keras/datasets/")
    os.system("mv cifar-10-batches-py.tar.gz ~/.keras/datasets/")
    main()
