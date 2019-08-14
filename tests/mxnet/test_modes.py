from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole.mxnet import SaveConfig, modes, reset_collections
from datetime import datetime
from tornasole.trials import create_trial

def test_modes():
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


