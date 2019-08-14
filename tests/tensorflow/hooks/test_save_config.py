from .utils import *
from tornasole.tensorflow import reset_collections
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
