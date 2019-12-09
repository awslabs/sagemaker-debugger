# Standard Library
import shutil
import subprocess
import sys
import uuid

# Third Party
import pytest
from tests.analysis.utils import delete_s3_prefix

# First Party
from smdebug.core.access_layer.utils import has_training_ended


@pytest.mark.slow  # 0:03 to run
def test_end_local_training():
    run_id = str(uuid.uuid4())
    out_dir = "/tmp/newlogsRunTest/" + run_id
    assert has_training_ended(out_dir) == False
    subprocess.check_call(
        [
            sys.executable,
            "examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py",
            "--output-uri",
            out_dir,
            "--num_steps",
            "10",
        ]
    )
    assert has_training_ended(out_dir)
    shutil.rmtree(out_dir)


@pytest.mark.slow  # 0:04 to run
def test_end_s3_training():
    run_id = str(uuid.uuid4())
    bucket = "smdebug-testing"
    key = f"outputs/{uuid.uuid4()}"
    out_dir = "s3://" + bucket + "/" + key
    assert has_training_ended(out_dir) == False
    subprocess.check_call(
        [
            sys.executable,
            "examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py",
            "--output-uri",
            out_dir,
            "--num_steps",
            "10",
        ]
    )
    assert has_training_ended(out_dir)
    delete_s3_prefix(bucket, key)
