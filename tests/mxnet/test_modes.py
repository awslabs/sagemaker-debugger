from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole.mxnet import SaveConfig, modes, reset_collections
from datetime import datetime
from tornasole.trials import create_trial

def test_modes(hook=None, path=None):
  if hook is None:
    reset_collections()
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    path = './newlogsRunTest/' + run_id
    hook = t_hook(out_dir=path,
                  save_config={modes.TRAIN: SaveConfig(save_interval=50),
                              modes.EVAL: SaveConfig(save_interval=10)})
  run_mnist_gluon_model(hook=hook, set_modes=True)

  tr = create_trial(path)
  assert len(tr.modes()) == 2
  assert len(tr.available_steps()) == 5
  assert len(tr.available_steps(mode=modes.TRAIN)) == 3
  assert len(tr.available_steps(mode=modes.EVAL)) == 2

def test_modes_hook_from_json_config():
  from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
  import shutil
  import os
  reset_collections()
  out_dir = 'newlogsRunTest2/test_modes_hookjson'
  shutil.rmtree(out_dir, True)
  os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/mxnet/test_json_configs/test_modes_hook.json'
  hook = t_hook.hook_from_config()
  test_modes(hook, out_dir)
  shutil.rmtree(out_dir, True)