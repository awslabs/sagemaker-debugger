# Third Party
import numpy as np
import pytest
import tensorflow as tf
from tests.tensorflow2.utils import is_tf_2_2

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.trials import create_trial


def get_data():
    images = np.zeros((64, 224))
    labels = np.zeros((64, 5))
    inputs = {"Image_input": images}
    outputs = {"output-softmax": labels}
    return inputs, outputs


def create_hook(trial_dir):
    hook = smd.KerasHook(trial_dir, save_all=True)
    return hook


def create_model():
    input_layer = tf.keras.layers.Input(name="Image_input", shape=(224), dtype="float32")
    model = tf.keras.layers.Dense(5)(input_layer)
    model = tf.keras.layers.Activation("softmax", name="output-softmax")(model)
    model = tf.keras.models.Model(inputs=input_layer, outputs=[model])
    return model


@pytest.mark.skipif(
    is_tf_2_2() is False,
    reason="Feature to save model inputs and outputs was first added for TF 2.2.0",
)
def test_support_dicts(out_dir):
    model = create_model()
    optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    inputs, labels = get_data()
    smdebug_hook = create_hook(out_dir)
    model.fit(inputs, labels, batch_size=16, epochs=10, callbacks=[smdebug_hook])
    model.save(out_dir, save_format="tf")
    trial = create_trial(out_dir)
    assert trial.tensor_names(collection=CollectionKeys.INPUTS) == ["inputs_0"]
    assert trial.tensor_names(collection=CollectionKeys.OUTPUTS) == ["labels_0", "predictions"]
