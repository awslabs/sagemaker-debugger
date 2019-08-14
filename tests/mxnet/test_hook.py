from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole import SaveConfig, modes
from tornasole.mxnet import reset_collections
from datetime import datetime
import shutil
from tornasole.core.access_layer.utils import has_training_ended

def test_hook():
  reset_collections()
  save_config = SaveConfig(save_steps=[0,1,2,3])
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  out_dir='./newlogsRunTest/' + run_id
  hook = t_hook(out_dir=out_dir, save_config=save_config)
  assert (has_training_ended(out_dir) == False)
  run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10)
  shutil.rmtree(out_dir)
