from tornasole_core.collection import Collection
from tornasole_core.collection_manager import CollectionManager
from tornasole_core.reduction_config import ReductionConfig

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
  cm.get('default').include('loss')
  cm.add(Collection('trial1'))
  cm.get('trial1').exclude('losses')
  cm.add('trial2')
  cm.get('trial2').include('total_loss')
  cm.export('cm.ts')
  cm2 = CollectionManager.load('cm.ts')
  assert cm == cm2

def test_manager():
  cm = CollectionManager()
  cm.get('default').include('loss')
  cm.add(Collection('trial1'))
  cm.get('trial1').exclude('losses')
  cm.add('trial2')
  cm.get('trial2').include('total_loss')
  assert len(cm.collections) == 3
  assert cm.get('default') == cm.collections['default']
  assert 'loss' in cm.get('default').include_regex
  assert 'losses' in cm.get('trial1').exclude_regex
  assert 'total_loss' in cm.collections['trial2'].include_regex