"""
This file is temporary, for testing with 2.X.
We'll need to integrate a more robust testing pipeline and make this part of pytest
before pushing to master.

This was tested with TensorFlow 2.1, by running
`python tests/tensorflow2/test_simple.py` from the main directory.
"""

# Standard Library
from tempfile import TemporaryDirectory

# Third Party
import tensorflow.compat.v2 as tf

# First Party
import smdebug.tensorflow as smd

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
cce = tf.keras.losses.CategoricalCrossentropy()

with TemporaryDirectory() as dirpath:
    hook = smd.KerasHook(out_dir=dirpath)

    # n_epochs = 1
    # batch_size = 32
    # for epoch in range(n_epochs):
    #     for data, labels in train_dataset.batch(batch_size).as_numpy_iterator():
    #         # batch is tuple of ((32,28,28), (32,))
    #         labels = tf.one_hot(labels, depth=10)
    #         with tf.GradientTape(persistent=True) as tape:
    #             logits = model(data) # (32,10)
    #             loss_value = cce(labels, logits)
    #             layer = model.layers[1]
    #             vars = layer

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, callbacks=[hook])
    model.evaluate(x_test, y_test, verbose=2, callbacks=[hook])

    trial = smd.create_trial(path=dirpath)
    print(hook)
    print(trial)
