from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole.mxnet import SaveConfig, Collection, reset_collections
import tornasole.mxnet as tm
import shutil

from datetime import datetime

def test_save_config():
  reset_collections()
  save_config_collection = SaveConfig(save_steps=[4,5,6])

  custom_collect = tm.get_collection("ReluActivation")
  custom_collect.set_save_config(save_config_collection)
  custom_collect.include(["relu*", "input_*", "output*"])
  save_config = SaveConfig(save_steps=[0,1,2,3])
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  out_dir = './newlogsRunTest/' + run_id
  hook = t_hook(out_dir=out_dir, save_config=save_config, include_collections=["ReluActivation", 'weights', 'bias','gradients', 'default'])
  run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10)
  shutil.rmtree(out_dir)

