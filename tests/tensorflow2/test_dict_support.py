# Third Party
import numpy as np
import tensorflow as tf

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.tensorflow import SaveConfig


def get_data():
    images = np.zeros((64, 224))
    labels = np.zeros((64, 5))
    inputs = {"Image_input": images}
    outputs = {"output-softmax": labels}
    return inputs, outputs


def create_model():
    input_layer = tf.keras.layers.Input(name="Image_input", shape=(224), dtype="float32")
    model = tf.keras.layers.Dense(5)(input_layer)
    model = tf.keras.layers.Activation("softmax", name="output-softmax")(model)
    model = tf.keras.models.Model(inputs=input_layer, outputs=[model])
    return model


def test_dict_support(out_dir):
    include_collection = [CollectionKeys.INPUTS, CollectionKeys.OUTPUTS]
    hook = smd.KerasHook(
        out_dir, save_config=SaveConfig(save_interval=1), include_collections=include_collection
    )
    hooks = [hook]

    model = create_model()
    optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    data = get_data()
    model.fit(data[0], data[1], batch_size=16, epochs=10, callbacks=hooks)
