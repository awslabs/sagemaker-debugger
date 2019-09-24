from tornasole.core.collection import Collection
from tornasole.core.collection_manager import CollectionManager, \
  COLLECTIONS_FILE_NAME
from tornasole.core.reduction_config import ReductionConfig

def test_export_load():
  # with none as save config
  c1 = Collection('default', include_regex=['conv2d'], reduction_config=ReductionConfig())
  c2 = Collection.load(c1.export())
  assert c1 == c2
  assert c1.export() == c2.export()

def test_load_empty():
  c = Collection('trial')
  assert c == Collection.load(c.export())

def test_manager_export_load():
  cm = CollectionManager()
  cm.create_collection('default')
  cm.get('default').include('loss')
  cm.add(Collection('trial1'))
  cm.add('trial2')
  cm.get('trial2').include('total_loss')
  cm.export(COLLECTIONS_FILE_NAME)
  cm2 = CollectionManager.load(COLLECTIONS_FILE_NAME)
  assert cm == cm2

def test_manager():
  cm = CollectionManager()
  cm.create_collection('default')
  cm.get('default').include('loss')
  cm.get('default').add_tensor_name('assaas')
  cm.add(Collection('trial1'))
  cm.add('trial2')
  cm.get('trial2').include('total_loss')
  assert len(cm.collections) == 3
  assert cm.get('default') == cm.collections['default']
  assert 'loss' in cm.get('default').include_regex
  assert len(cm.get('default').get_tensor_names()) > 0
  assert 'total_loss' in cm.collections['trial2'].include_regex
