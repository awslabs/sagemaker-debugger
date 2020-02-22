# Standard Library
import os

# Third Party
import tensorflow as tf

# First Party
from smdebug.core.utils import get_path_to_collections
from smdebug.tensorflow.collection import Collection, CollectionManager


def test_manager_export_load(out_dir):
    cm = CollectionManager()
    cm.get("default").include("loss")
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rc = tf.math.reduce_max(c)
    cm.get("default").add_tensor(c)
    cm.add(Collection("trial1"))
    cm.add("trial2")
    cm.get("trial2").include("total_loss")
    cm.export(out_dir, "cm.json")
    cm2 = CollectionManager.load(os.path.join(get_path_to_collections(out_dir), "cm.json"))
    assert cm == cm2


def test_add_variable():
    cm = CollectionManager()
    tf.reset_default_graph()
    var = tf.Variable(tf.zeros([1.0, 2.0, 3.0]))
    cm.get("test").add(var)
    # this works here, and works for tf session
    # but in keras each time value is called, it increases readVariableOp counter
    assert var.value().name in cm.get("test").get_tensors_dict()
    assert var.name in cm.get("test").tensor_names
    assert cm.get("test").get_tensor(var.value().name).original_tensor == var


def test_add_variable_with_name():
    cm = CollectionManager()
    tf.reset_default_graph()
    var = tf.Variable(tf.zeros([1.0, 2.0, 3.0]))
    cm.get("test").add_variable(var, export_name="zeros_var")
    # this works here, and works for tf session
    # but in keras each time value is called, it increases readVariableOp counter
    assert var.value().name in cm.get("test").get_tensors_dict()
    assert "zeros_var" in cm.get("test").tensor_names
    assert cm.get("test").get_tensor(var.value().name).original_tensor == var
