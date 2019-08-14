import pytest
from tornasole.core.save_config import SaveConfig

def test_export_load():
  r1 = SaveConfig(save_interval=11, skip_num_steps=10, save_steps=[50], when_nan=['loss:0'])
  r2 = SaveConfig.load(r1.export())
  assert r1.export() == r2.export()
  assert r1 == r2

def test_load_empty():
  r1 = SaveConfig()
  assert r1 == SaveConfig.load(r1.export())
