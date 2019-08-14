from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole.mxnet import Collection, reset_collections
from tornasole import SaveConfig
from tornasole.trials import create_trial
import tornasole.mxnet as tm
from datetime import datetime
import numpy as np
import shutil

def test_hook_all_zero():
  reset_collections()
  tm.get_collection('ReluActivation').include(["relu*", "input_*"])
  save_config = SaveConfig(save_steps=[0,1,2,3])
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  out_dir = './newlogsRunTest/' + run_id
  print("Registering the hook with out_dir {0}".format(out_dir))
  hook = t_hook(out_dir=out_dir, save_config=save_config, include_collections=['ReluActivation','weights', 'bias','gradients'])
  run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10, make_input_zero=True)


  print("Created the trial with out_dir {0}".format(out_dir))
  tr = create_trial(out_dir)
  assert tr
  assert len(tr.available_steps()) == 4

  tnames = tr.tensors_matching_regex('conv._input')
  print(tnames)
  tname = tr.tensors_matching_regex('conv._input')[0]
  print(tname)
  print(tr.tensor(tname).steps())
  conv_tensor_value = tr.tensor(tname).value(step_num=0)
  is_zero = np.all(conv_tensor_value==0)
  assert is_zero == True

  shutil.rmtree(out_dir)