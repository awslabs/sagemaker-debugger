# Third Party
import tensorflow as tf

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.core.modes import ModeKeys
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


def helper_create_hook(out_dir, collections, include_regex=None):
    hook = smd.KerasHook(
        out_dir, save_config=SaveConfig(save_interval=3), include_collections=collections
    )

    if include_regex:
        for collection in collections:
            hook.get_collection(collection).include(include_regex)

    hook.register_model(model)
    hook.set_mode(ModeKeys.TRAIN)
    hook._prepare_collections()
    hook._increment_step()
    hook.on_train_begin()
    return hook


def test_should_save_tensor_with_default_collections(out_dir):
    hook = helper_create_hook(out_dir, TF_DEFAULT_SAVED_COLLECTIONS)
    for layer in model.layers:
        layer_name = layer.name
        assert not hook.should_save_tensor_or_collection(layer_name, CollectionKeys.GRADIENTS)
        assert not hook.should_save_tensor_or_collection(layer_name, CollectionKeys.LAYERS)


def test_should_save_tensor_with_tf_collection(out_dir):
    hook = helper_create_hook(out_dir, [CollectionKeys.GRADIENTS, CollectionKeys.LAYERS])
    for layer in model.layers:
        layer_name = layer.name
        assert hook.should_save_tensor_or_collection(layer_name, CollectionKeys.GRADIENTS)
        assert hook.should_save_tensor_or_collection(layer_name, CollectionKeys.LAYERS)


def test_should_save_tensor_with_custom_collection(out_dir):
    hook = helper_create_hook(out_dir, ["custom_coll"], include_regex="dense")
    for layer in model.layers:
        layer_name = layer.name
        if "dense" in layer_name:
            assert hook.should_save_tensor_or_collection(layer_name, CollectionKeys.GRADIENTS)
            assert hook.should_save_tensor_or_collection(layer_name, CollectionKeys.LAYERS)
        else:
            assert not hook.should_save_tensor_or_collection(layer_name, CollectionKeys.GRADIENTS)
            assert not hook.should_save_tensor_or_collection(layer_name, CollectionKeys.LAYERS)


def test_should_save_tensor_behavior_without_prepare_collections(out_dir):
    """Always return false if an attempt to save a tensor is made before the collections are prepared.
    This can happen if the fn is called before callbacks are init."""
    hook = smd.KerasHook(out_dir, save_config=SaveConfig(save_interval=3), save_all=True)
    assert not hook.should_save_tensor_or_collection("dummy", CollectionKeys.GRADIENTS)
    assert not hook.should_save_tensor_or_collection("dummy", CollectionKeys.LAYERS)
