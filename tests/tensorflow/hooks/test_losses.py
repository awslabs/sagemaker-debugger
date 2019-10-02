from .utils import *
import tornasole.tensorflow as ts
from tornasole.trials import create_trial
import shutil
import pytest

from .test_estimator_modes import help_test_mnist

@pytest.mark.slow # 0:02 to run
def test_mnist_local():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
  help_test_mnist(trial_dir, ts.SaveConfig(save_interval=2))
  tr = create_trial(trial_dir)
  assert len(tr.collection('losses').get_tensor_names()) == 1
  for t in tr.collection('losses').get_tensor_names():
    assert len(tr.tensor(t).steps()) == 4
  shutil.rmtree(trial_dir)
