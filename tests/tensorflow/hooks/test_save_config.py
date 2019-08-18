from .utils import *
from tests.tensorflow.hooks.test_estimator_modes import help_test_mnist
from tornasole.tensorflow import reset_collections, get_collection, TornasoleHook, modes
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
import shutil

def test_save_config():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
  tf.reset_default_graph()
  reset_collections()

  hook = TornasoleHook(out_dir=trial_dir,
                       save_all=False,
                       save_config=SaveConfig(save_interval=2))
  simple_model(hook)
  _, files = get_dirs_files(trial_dir)
  steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))
  assert len(steps) == 5
  assert len(files) == 1

def test_save_config_skip_steps():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)

  tf.reset_default_graph()
  reset_collections()

  hook = TornasoleHook(out_dir=trial_dir,
                       save_all=False,
                       save_config=SaveConfig(save_interval=2, skip_num_steps=8))
  simple_model(hook, steps=20)
  _, files = get_dirs_files(trial_dir)
  steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))
  assert len(steps) == 6

  shutil.rmtree(trial_dir)

def test_save_config_modes():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)

  tf.reset_default_graph()
  reset_collections()

  get_collection('weights').set_save_config({modes.TRAIN: SaveConfig(save_interval=1),
                                            modes.EVAL: SaveConfig(save_interval=5)})
  hook = TornasoleHook(out_dir=trial_dir)
  tr = help_test_mnist(trial_dir, hook=hook)
  for tname in tr.tensors_in_collection('weights'):
    t = tr.tensor(tname)
    assert len(t.steps(mode=modes.TRAIN)) == 30
    assert len(t.steps(mode=modes.EVAL)) == 16

  shutil.rmtree(trial_dir)

def test_save_config_modes_json():
  out_dir = 'newlogsRunTest1/test_save_config_modes_config_coll'
  shutil.rmtree(out_dir, ignore_errors=True)
  os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/tensorflow/hooks/test_json_configs/test_save_config_modes_config_coll.json'
  reset_collections()
  hook = TornasoleHook.hook_from_config()
  tr = help_test_mnist(out_dir, hook=hook)
  for tname in tr.tensors_in_collection('weights'):
    t = tr.tensor(tname)
    assert len(t.steps(mode=modes.TRAIN)) == 30
    assert len(t.steps(mode=modes.EVAL)) == 16
  shutil.rmtree(out_dir, ignore_errors=True)

  # get_collection('weights').set_save_config({modes.TRAIN: SaveConfig(save_interval=1),
  #                                           modes.EVAL: SaveConfig(save_interval=5)})
  # hook = TornasoleHook(out_dir=trial_dir)
  # tr = help_test_mnist(trial_dir, hook=hook)
