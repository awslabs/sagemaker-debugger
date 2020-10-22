# Third Party
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# First Party
import smdebug.tensorflow as smd
from smdebug.tensorflow.utils import does_tf_support_mixed_precision_training
from smdebug.trials import create_trial

# Test Reference: https://github.com/tensorflow/docs/blob/master/site/en/guide/mixed_precision.ipynb


@pytest.mark.skipif(
    does_tf_support_mixed_precision_training() is False,
    reason="The Keras mixed precision API is first available in TensorFlow 2.1.0",
)
def test_mixed_precision_training(out_dir):

    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    hook = smd.KerasHook(out_dir=out_dir, save_all=True)
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)

    inputs = keras.Input(shape=(784,), name="digits")
    if tf.config.list_physical_devices("GPU"):
        # The model will run with 4096 units on a GPU
        num_units = 4096
    else:
        # Use fewer units on CPUs so the model finishes in a reasonable amount of time
        # The model will run with 64 units on a CPU
        num_units = 64
    dense1 = layers.Dense(num_units, activation="relu", name="dense_1")
    x = dense1(inputs)
    dense2 = layers.Dense(num_units, activation="relu", name="dense_2")
    x = dense2(x)

    # CORRECT: softmax and model output are float32
    x = layers.Dense(10, name="dense_logits")(x)
    outputs = layers.Activation("softmax", dtype="float32", name="predictions")(x)

    # The linear activation is an identity function. So this simply casts 'outputs'
    # to float32. In this particular case, 'outputs' is already float32 so this is a
    # no-op.
    outputs = layers.Activation("linear", dtype="float32")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    initial_weights = model.get_weights()

    hooks = [hook]
    history = model.fit(
        x_train, y_train, batch_size=8192, epochs=5, callbacks=hooks, validation_split=0.2
    )
    test_scores = model.evaluate(x_test, y_test, verbose=2)

    trial = create_trial(out_dir)
    assert len(trial.tensor_names()) == 30
