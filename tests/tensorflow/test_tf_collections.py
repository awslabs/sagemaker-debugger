# Standard Library
import os
import uuid

# Third Party
import tensorflow as tf

# First Party
from smdebug.core.utils import get_path_to_collections
from smdebug.tensorflow import Collection, CollectionManager, get_collection, reset_collections
from smdebug.tensorflow.tensor_ref import get_tf_names


def test_manager_export_load(out_dir):
    cm = CollectionManager()
    cm.get("default").include("loss")
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rc = tf.math.reduce_max(c)
    cm.get("default").add_tensor(c)
    cm.get("default").add_reduction_tensor(rc, c)
    cm.add(Collection("trial1"))
    cm.add("trial2")
    cm.get("trial2").include("total_loss")
    cm.export(out_dir, "cm.json")
    cm2 = CollectionManager.load(os.path.join(get_path_to_collections(out_dir), "cm.json"))
    assert cm == cm2


def test_add_variable():
    reset_collections()
    tf.reset_default_graph()
    var = tf.Variable(tf.zeros([1.0, 2.0, 3.0]))
    get_collection("test").add(var)
    assert get_tf_names(var)[0] in get_collection("test").get_tensors_dict()
    assert var.name in get_collection("test").tensor_names
    assert get_collection("test").get_tensor(get_tf_names(var)[0]).original_tensor == var


def test_add_variable_with_name():
    reset_collections()
    tf.reset_default_graph()
    var = tf.Variable(tf.zeros([1.0, 2.0, 3.0]))
    get_collection("test").add_variable(var, export_name="zeros_var")
    assert get_tf_names(var)[0] in get_collection("test").get_tensors_dict()
    assert "zeros_var" in get_collection("test").tensor_names
    assert get_collection("test").get_tensor(get_tf_names(var)[0]).original_tensor == var
