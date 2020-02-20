# Standard Library
import shutil

# Third Party
import pytest

# First Party
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.access_layer.utils import (
    delete_s3_prefixes,
    has_training_ended,
    training_has_ended,
)
from smdebug.core.utils import ensure_dir, is_s3


def test_local_training_end():
    localdir = "/tmp/training_end_test_dir"
    ensure_dir(localdir, is_file=False)
    training_has_ended(localdir)
    assert has_training_ended(localdir) is True
    shutil.rmtree(localdir)


def test_negative_local_training_end():
    localdir = "/tmp/training_end_test_dir_negative"
    assert has_training_ended(localdir) is False


@pytest.mark.slow  # 0:04 to run
def test_s3_training_end():
    s3dir = "s3://smdebugcodebuildtest/training_end_test_dir"
    _, bucket, key = is_s3(s3dir)
    f = TSAccessS3(bucket_name=bucket, key_name=key)
    f.close()
    training_has_ended(s3dir)
    assert has_training_ended(s3dir) is True
    delete_s3_prefixes(bucket, key)


@pytest.mark.slow  # 0:05 to run
def test_negative_s3_training_end():
    s3dir = "s3://smdebugcodebuildtest/training_end_test_dir_negative"
    assert has_training_ended(s3dir) is False
