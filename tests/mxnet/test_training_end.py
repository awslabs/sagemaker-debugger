import shutil
from tornasole.core.access_layer.utils import has_training_ended
import subprocess
import uuid
from tests.analysis.utils import delete_s3_prefix
import sys
import pytest


@pytest.mark.slow # 0:03 to run
def test_end_local_training():
  run_id = str(uuid.uuid4())
  out_dir='./newlogsRunTest/' + run_id
  assert has_training_ended(out_dir) == False
  subprocess.check_call([sys.executable, "examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py",
                         "--output-uri", out_dir, '--num_steps', '10'])
  assert has_training_ended(out_dir)
  shutil.rmtree(out_dir)


@pytest.mark.slow # 0:04 to run
def test_end_s3_training():
  run_id = str(uuid.uuid4())
  bucket = 'tornasolecodebuildtest'
  key = 'newlogsRunTest/' + run_id
  out_dir= bucket + "/" + key
  assert has_training_ended(out_dir) == False
  subprocess.check_call([sys.executable, "examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py",
                         "--output-uri", out_dir, '--num_steps', '10'])
  assert has_training_ended(out_dir)
  delete_s3_prefix(bucket, key)
