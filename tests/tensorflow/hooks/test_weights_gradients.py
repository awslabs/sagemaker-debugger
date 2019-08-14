from .utils import *
from tornasole.tensorflow import reset_collections

def test_only_w_g():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)

  tf.reset_default_graph()
  reset_collections()

  hook = TornasoleHook(out_dir=trial_dir,
                       save_all=False, save_config=SaveConfig(save_interval=2))
  simple_model(hook)
  steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))
  _, files = get_dirs_files(trial_dir)

  assert 'collections.ts' in files
  cm = CollectionManager.load(join(trial_dir, 'collections.ts'))
  num_tensors_loaded_collection = len(cm.collections['weights'].tensor_names) + \
                                  len(cm.collections['gradients'].tensor_names) + \
                                  len(cm.collections['default'].tensor_names)
  assert num_tensors_loaded_collection == 2
  assert len(steps) == 5
  # for step in steps:
  #   i = 0
  #   filepath, size = get_event_file_path_length(join(rank_dir, step))
  #   for (n, t) in get_tensors_from_event_file(filepath):
  #     i += 1
  #   assert i == 2
