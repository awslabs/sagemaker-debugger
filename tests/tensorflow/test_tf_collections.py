from tornasole.tensorflow import Collection, CollectionManager
from tornasole.tensorflow import add_to_collection, get_collection
import tensorflow as tf

def test_manager_export_load():
  cm = CollectionManager()
  cm.get('default').include('loss')
  c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  rc = tf.math.reduce_max(c)
  cm.get('default').add_tensor(c)
  cm.get('default').add_reduction_tensor(rc, c)
  cm.add(Collection('trial1'))
  cm.add('trial2')
  cm.get('trial2').include('total_loss')
  cm.export('cm.ts')
  cm2 = CollectionManager.load('cm.ts')
  assert cm == cm2
