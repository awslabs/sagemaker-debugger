from .utils import *
from tornasole.core.modes import ModeKeys
import glob, shutil
from tornasole.core.reader import FileReader
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR


def test_collection_defaults_json():
    trial_dir = "newlogsRunTest/test_collection_defaults_json"
    shutil.rmtree(trial_dir, ignore_errors=True)
    pre_test_clean_up()
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = "tests/tensorflow/hooks/test_json_configs/test_collection_defaults.json"
    hook = TornasoleHook.hook_from_config()
    # Check save_intervals for each mode
    assert hook.save_manager.default_save_modes.get_save_config(ModeKeys.TRAIN).save_interval == 2
    assert hook.save_manager.default_save_modes.get_save_config(ModeKeys.EVAL).save_interval == 3
    assert hook.save_manager.default_save_modes.get_save_config(ModeKeys.PREDICT).save_interval == 1
    assert hook.save_manager.default_save_modes.get_save_config(ModeKeys.GLOBAL).save_interval == 1
    # Check include_collections
    assert 'weights' in hook.include_collections and 'losses' in hook.include_collections
    assert len(hook.include_collections) == 2
    # Check collection configurations for losses
    assert hook.save_manager.configs_for_collections == {} #TODO(nieljare): Is this used?
    assert hook.save_manager.collection_manager.collections['losses'].save_config[ModeKeys.TRAIN].save_interval == 2
    assert hook.save_manager.collection_manager.collections['losses'].save_config[ModeKeys.EVAL].save_interval == 4
    assert hook.save_manager.collection_manager.collections['losses'].save_config[ModeKeys.PREDICT].save_interval == 1
    assert hook.save_manager.collection_manager.collections['losses'].save_config[ModeKeys.GLOBAL].save_interval == 5
    # Check collection configuration for weights
    assert hook.save_manager.collection_manager.collections['weights'].save_config[ModeKeys.TRAIN].save_interval == 2
    assert hook.save_manager.collection_manager.collections['weights'].save_config[ModeKeys.EVAL].save_interval == 3
    assert hook.save_manager.collection_manager.collections['weights'].save_config[ModeKeys.PREDICT].save_interval == 1
    assert hook.save_manager.collection_manager.collections['weights'].save_config[ModeKeys.GLOBAL].save_interval == 1
