# Standard Library
import argparse

# Third Party
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# First Party
import smdebug.tensorflow as smd
from smdebug.tensorflow import SaveConfig


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
    zcc=False,
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

    if hook is None and zcc is False:
        if save_config is None:
            save_config = SaveConfig(save_interval=3)

        hook = smd.KerasHook(
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

    if zcc is False:
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/tmp/run")
    parser.add_argument("--include_workers", type=str, default="one")
    parser.add_argument("--zcc", type=str2bool, default=False)
    args = parser.parse_args()
    strategy = train_model(
        args.out_dir,
        include_collections=None,
        save_all=True,
        steps=["train", "eval", "predict", "train"],
        include_workers=args.include_workers,
        eager=True,
        zcc=args.zcc,
    )
