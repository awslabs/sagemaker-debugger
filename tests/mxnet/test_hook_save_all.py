from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole.mxnet import SaveConfig, reset_collections
import shutil
from datetime import datetime

def test_save_all():
  reset_collections()
  save_config = SaveConfig(save_steps=[0,1,2,3])
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  out_dir = './newlogsRunTest/' + run_id
  hook = t_hook(out_dir=out_dir, save_config=save_config, save_all=True)
  run_mnist_gluon_model(hook=hook, num_steps_train=7, num_steps_eval=5)
  shutil.rmtree(out_dir)

