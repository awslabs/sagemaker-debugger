# Third Party

# First Party
from smdebug import modes
from smdebug.core.save_config import SaveConfig


def test_export_load():
    r1 = SaveConfig(save_interval=11, start_step=10, save_steps=[50])
    r2 = SaveConfig.from_json(r1.to_json())
    assert r1.to_json() == r2.to_json()
    assert r1 == r2


def test_load_empty():
    r1 = SaveConfig()
    assert r1 == SaveConfig.from_json(r1.to_json())


def test_load_none():
    r1 = SaveConfig(start_step=100)
    assert r1 == SaveConfig.from_json(r1.to_json())


def test_end_step():
    s = SaveConfig(end_step=0)
    assert s.should_save_step(modes.GLOBAL, 0) is False
    assert s.should_save_step(modes.GLOBAL, 19) is False
    assert s.should_save_step(modes.GLOBAL, 100) is False
