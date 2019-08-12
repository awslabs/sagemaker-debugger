import boto3 as boto3
from tornasole_core.access_layer.utils import has_training_ended
from tornasole_core.access_layer.utils import training_has_ended
import shutil
from tornasole_core.utils import is_s3
from tornasole_core.access_layer.file import ensure_dir

def del_s3(bucket,file_path):
    s3_client = boto3.client('s3')
    s3_client.delete_object(Bucket=bucket, Key=file_path)

def test_local_training_end():
    localdir = "./training_end_test_dir"
    training_has_ended(localdir)
    assert has_training_ended(localdir) == True
    shutil.rmtree(localdir)

def test_negative_local_training_end():
    localdir = "./training_end_test_dir_negative"
    assert has_training_ended(localdir) == False

def test_s3_training_end():
    s3dir = 's3://tornasolecodebuildtest/training_end_test_dir'
    bucket = 'tornasolecodebuildtest'
    training_has_ended(s3dir)
    assert has_training_ended(s3dir) == True
    del_s3(bucket, s3dir)

def test_negative_s3_training_end():
    s3dir = 's3://tornasolecodebuildtest/training_end_test_dir_negative'
    assert has_training_ended(s3dir) == False