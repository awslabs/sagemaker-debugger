from tornasole.tensorflow import Collection, CollectionManager
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
  cm.export('cm.json')
  cm2 = CollectionManager.load('cm.json')
  assert cm == cm2
