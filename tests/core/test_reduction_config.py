import pytest
from tornasole.core.reduction_config import ReductionConfig

def test_export_load():
  r1 = ReductionConfig(only_shape=True, reductions=['min'], norms=['l2'])
  r2 = ReductionConfig.load(r1.export())
  assert r1 == r2
  assert r1.export() == r2.export()

def test_load_empty():
  r1 = ReductionConfig()
  assert r1 == ReductionConfig.load(r1.export())