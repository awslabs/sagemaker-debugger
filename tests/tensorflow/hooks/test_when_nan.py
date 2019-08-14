from .utils import *
from tornasole.tensorflow import reset_collections

def test_when_nan():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)

  tf.reset_default_graph()
  tf.set_random_seed(1)
  np.random.seed(1)
  reset_collections()

  hook = TornasoleHook(out_dir=trial_dir,
                       save_config=SaveConfig(save_interval=10, when_nan=['loss:0']))
  simple_model(hook, steps=100, lr=4e20)
  steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))
  _, files = get_dirs_files(trial_dir)

  assert 'collections.ts' in files
  cm = CollectionManager.load(join(trial_dir, 'collections.ts'))
  num_tensors_loaded_collection = len(cm.collections['weights'].tensor_names) + \
                                  len(cm.collections['gradients'].tensor_names) + \
                                  len(cm.collections['when_nan'].tensor_names) + \
                                  len(cm.collections['default'].tensor_names)
  assert num_tensors_loaded_collection == 3

  num_steps_with_files = 0
  for step in steps:
    filepath, size = get_event_file_path_length(join(trial_dir, 'events', step))
    if size > 0:
      num_steps_with_files += 1
  assert num_steps_with_files == 35
