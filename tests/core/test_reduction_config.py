# Third Party

# First Party
from smdebug.core.reduction_config import ReductionConfig


def test_export_load():
    r1 = ReductionConfig(only_shape=True, reductions=["min"], norms=["l2"])
    r2 = ReductionConfig.from_json(r1.to_json())
    assert r1 == r2
    assert r1.to_json() == r2.to_json()


def test_load_empty():
    r1 = ReductionConfig()
    assert r1 == ReductionConfig.from_json(r1.to_json())


def test_load_none():
    r1 = ReductionConfig(save_raw_tensor=True)
    assert r1 == ReductionConfig.from_json(r1.to_json())
