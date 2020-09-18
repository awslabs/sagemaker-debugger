# Third Party
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D

# First Party
import smdebug.tensorflow as smd


def create_dataset():
    # Download and load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data("MNIST-data")
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000, seed=123).batch(2)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(2)

    return train_ds, test_ds


def test_functional_model(out_dir, tf_eager_mode):
    if tf_eager_mode is False:
        tf.compat.v1.disable_eager_execution()
    else:
        return
    num_classes = 10
    train_ds, test_ds = create_dataset()

    # Input image dimensions
    img_rows, img_cols = 28, 28

    img_inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation="relu")(img_inputs)
    x1 = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x1)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=img_inputs, outputs=out)

    smd_callback = smd.KerasHook(
        export_tensorboard=False, out_dir=out_dir, include_collections=["custom"]
    )

    smd_callback.get_collection("custom").add_for_mode([x1], mode=smd.modes.TRAIN)
    smd_callback.save_config = smd.SaveConfig(save_interval=1)
    opt = tf.keras.optimizers.Adadelta(1.0)

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=opt,
        experimental_run_tf_function=False,
    )

    callbacks = [smd_callback]
    model.fit(train_ds, epochs=1, steps_per_epoch=100, callbacks=callbacks)

    trial = smd.create_trial(out_dir)
    assert len(trial.tensor_names(collection="custom")) == 1
