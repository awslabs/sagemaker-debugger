# Third Party
import tensorflow as tf

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.tensorflow import SaveConfig
from smdebug.tensorflow.constants import TF_DEFAULT_SAVED_COLLECTIONS

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


def test_should_save_tensor_with_default_collections(out_dir):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=TF_DEFAULT_SAVED_COLLECTIONS,
    )
    hook.register_model(model)
    hook.on_train_begin()
    for layer in model.layers:
        layer_name = layer.name
        assert not hook.should_save_tensor(layer_name, CollectionKeys.GRADIENTS)
        assert not hook.should_save_tensor(layer_name, CollectionKeys.LAYERS)


def test_should_save_tensor_with_tf_collection(out_dir):
    hook = smd.KerasHook(
        out_dir,
        save_config=SaveConfig(save_interval=3),
        include_collections=[CollectionKeys.LAYERS, CollectionKeys.GRADIENTS],
    )
    hook.register_model(model)
    hook.on_train_begin()
    for layer in model.layers:
        layer_name = layer.name
        assert hook.should_save_tensor(layer_name, CollectionKeys.GRADIENTS)
        assert hook.should_save_tensor(layer_name, CollectionKeys.LAYERS)


def test_should_save_tensor_with_custom_collection(out_dir):
    hook = smd.KerasHook(
        out_dir, save_config=SaveConfig(save_interval=3), include_collections=["custom_coll"]
    )
    hook.get_collection("custom_coll").include("dense")
    hook.register_model(model)
    hook.on_train_begin()
    for layer in model.layers:
        layer_name = layer.name
        if "dense" in layer_name:
            assert hook.should_save_tensor(layer_name, CollectionKeys.GRADIENTS)
            assert hook.should_save_tensor(layer_name, CollectionKeys.LAYERS)
        else:
            assert not hook.should_save_tensor(layer_name, CollectionKeys.GRADIENTS)
            assert not hook.should_save_tensor(layer_name, CollectionKeys.LAYERS)
