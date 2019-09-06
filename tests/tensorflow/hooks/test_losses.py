from .utils import *
import tornasole.tensorflow as ts
import shutil

from .test_estimator_modes import help_test_mnist

def test_mnist_local():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
  tr = help_test_mnist(trial_dir, ts.SaveConfig(save_interval=2))
  assert len(tr.collection('losses').get_tensor_names()) == 1
  for t in tr.collection('losses').get_tensor_names():
    assert len(tr.tensor(t).steps()) == 55
  shutil.rmtree(trial_dir)