from tornasole.tensorflow import Collection, CollectionManager
from tornasole.core.utils import get_path_to_collections
import tensorflow as tf

import uuid
import os


def test_manager_export_load():
    id = str(uuid.uuid4())
    path = "/tmp/tests/" + id
    cm = CollectionManager()
    cm.get("default").include("loss")
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rc = tf.math.reduce_max(c)
    cm.get("default").add_tensor(c)
    cm.get("default").add_reduction_tensor(rc, c)
    cm.add(Collection("trial1"))
    cm.add("trial2")
    cm.get("trial2").include("total_loss")
    cm.export(path, "cm.json")
    cm2 = CollectionManager.load(os.path.join(get_path_to_collections(path), "cm.json"))
    assert cm == cm2
