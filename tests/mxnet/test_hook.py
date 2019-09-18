from .mnist_gluon_model import run_mnist_gluon_model
from tornasole.mxnet.hook import TornasoleHook as t_hook
from tornasole import SaveConfig
from tornasole.mxnet import reset_collections
from datetime import datetime
import shutil
from tornasole.core.access_layer.utils import has_training_ended
import os
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR, DEFAULT_SAGEMAKER_TORNASOLE_PATH


def test_hook():
  reset_collections()
  save_config = SaveConfig(save_steps=[0,1,2,3])
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  out_dir='newlogsRunTest/' + run_id
  hook = t_hook(out_dir=out_dir, save_config=save_config)
  assert (has_training_ended(out_dir) == False)
  run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True)
  shutil.rmtree(out_dir)

def test_hook_from_json_config():
  reset_collections()
  out_dir = 'newlogsRunTest1/test_hook_from_json_config'
  shutil.rmtree(out_dir, True)
  os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/mxnet/test_json_configs/test_hook_from_json_config.json'
  hook = t_hook.hook_from_config()
  assert (has_training_ended(out_dir) == False)
  run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True)
  shutil.rmtree(out_dir, True)

def test_hook_from_json_config_full():
  reset_collections()
  out_dir = 'newlogsRunTest2/test_hook_from_json_config_full'
  shutil.rmtree(out_dir, True)
  os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/mxnet/test_json_configs/test_hook_from_json_config_full.json'
  hook = t_hook.hook_from_config()
  assert (has_training_ended(out_dir) == False)
  run_mnist_gluon_model(hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True)
  shutil.rmtree(out_dir, True)

def test_default_hook():
  reset_collections()
  if TORNASOLE_CONFIG_FILE_PATH_ENV_STR in os.environ:
    del os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR]
  hook = t_hook.hook_from_config()
  assert(hook.out_dir == DEFAULT_SAGEMAKER_TORNASOLE_PATH)



