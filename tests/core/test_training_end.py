from tornasole.core.access_layer.utils import (
    has_training_ended,
    training_has_ended,
    delete_s3_prefixes,
)
import pytest
import shutil
from tornasole.core.utils import is_s3
from tornasole.core.access_layer.file import ensure_dir
from tornasole.core.access_layer.s3 import TSAccessS3


def test_local_training_end():
    localdir = "/tmp/training_end_test_dir"
    ensure_dir(localdir, is_file=False)
    training_has_ended(localdir)
    assert has_training_ended(localdir) == True
    shutil.rmtree(localdir)


def test_negative_local_training_end():
    localdir = "/tmp/training_end_test_dir_negative"
    assert has_training_ended(localdir) == False


@pytest.mark.slow  # 0:04 to run
def test_s3_training_end():
    s3dir = "s3://tornasolecodebuildtest/training_end_test_dir"
    _, bucket, key = is_s3(s3dir)
    f = TSAccessS3(bucket_name=bucket, key_name=key)
    f.close()
    training_has_ended(s3dir)
    assert has_training_ended(s3dir) == True
    delete_s3_prefixes(bucket, key)


@pytest.mark.slow  # 0:05 to run
def test_negative_s3_training_end():
    s3dir = "s3://tornasolecodebuildtest/training_end_test_dir_negative"
    assert has_training_ended(s3dir) == False
