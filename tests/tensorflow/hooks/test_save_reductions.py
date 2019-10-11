from .utils import *
from tornasole.tensorflow import reset_collections, get_collections, CollectionManager
import shutil
import glob
from tornasole.core.reader import FileReader
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
from tornasole.core.config_constants import TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME

def helper_save_reductions(trial_dir, hook):
  simple_model(hook)
  _, files = get_dirs_files(trial_dir)
  coll = get_collections()
  assert len(coll['weights'].tensor_names) == 1
  assert len(coll['gradients'].tensor_names) == 1

  assert TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME in files
  cm = CollectionManager.load(join(trial_dir, TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME))
  assert len(cm.collections) == len(coll)
  assert len(cm.collections['weights'].tensor_names) == 1
  assert len(cm.collections['gradients'].tensor_names) == 1
  # as we hadn't asked to be saved
  assert len(cm.collections['optimizer_variables'].tensor_names) == 0
  assert len(cm.collections['default'].tensor_names) == 0
  num_tensors_loaded_collection = len(cm.collections['weights'].tensor_names) + \
                                  len(cm.collections['gradients'].tensor_names) + \
                                  len(cm.collections['default'].tensor_names)
  num_tensors_collection = len(coll['weights'].tensor_names) + \
                           len(coll['gradients'].tensor_names) + \
                           len(coll['default'].tensor_names)
  assert num_tensors_collection == num_tensors_loaded_collection
  steps, _ = get_dirs_files(os.path.join(trial_dir, 'events'))
  assert len(steps) == 10
  for step in steps:
    i = 0
    size = 0
    fs = glob.glob(join(trial_dir, 'events', step, '**', '*.tfevents'), recursive=True)
    for f in fs:
      fr = FileReader(f)
      for x in fr.read_tensors():
        tensor_name, step, tensor_data, mode, mode_step = x
        i += 1
        size += tensor_data.nbytes if tensor_data is not None else 0
    assert i == 48
    assert size == 192

  shutil.rmtree(trial_dir)


def test_save_reductions():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
  pre_test_clean_up()
  rdnc = ReductionConfig(reductions=['min', 'max', 'mean', 'prod', 'std', 'sum', 'variance'],
                         abs_reductions=['min', 'max', 'mean', 'prod', 'std', 'sum', 'variance'],
                         norms=['l1', 'l2'])
  hook = TornasoleHook(out_dir=trial_dir,
                       save_config=SaveConfig(save_interval=1),
                       reduction_config=rdnc)
  helper_save_reductions(trial_dir, hook)


def test_save_reductions_json():
  trial_dir = "newlogsRunTest1/test_save_reductions"
  shutil.rmtree(trial_dir, ignore_errors=True)
  pre_test_clean_up()
  os.environ[
    TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = "tests/tensorflow/hooks/test_json_configs/test_save_reductions.json"
  hook = TornasoleHook.hook_from_config()
  helper_save_reductions(trial_dir, hook)
