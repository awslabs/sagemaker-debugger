import shutil
from tornasole.core.access_layer.utils import has_training_ended
import subprocess
import uuid
import boto3
import sys


def test_end_local_training():
  run_id = str(uuid.uuid4())
  out_dir='./newlogsRunTest/' + run_id
  assert has_training_ended(out_dir) == False
  subprocess.check_call([sys.executable, "examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py",
                         "--output-uri", out_dir, '--num_steps', '10'])
  assert has_training_ended(out_dir)
  shutil.rmtree(out_dir)


def del_s3(bucket,file_path):
    s3_client = boto3.client('s3')
    s3_client.delete_object(Bucket=bucket, Key=file_path)


def test_end_s3_training():
  run_id = str(uuid.uuid4())
  bucket = 'tornasolecodebuildtest'
  key = 'newlogsRunTest/' + run_id
  out_dir= bucket + "/" + key
  assert has_training_ended(out_dir) == False
  subprocess.check_call([sys.executable, "examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py",
                         "--output-uri", out_dir, '--num_steps', '10'])
  assert has_training_ended(out_dir)
  del_s3(bucket, key)
