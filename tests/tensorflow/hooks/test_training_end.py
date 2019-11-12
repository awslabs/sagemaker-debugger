# Standard Library
import os
import shutil
import subprocess
import sys

# Third Party
import pytest
import tensorflow as tf

# First Party
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.tensorflow import reset_collections

# Local
from .utils import *


@pytest.mark.slow  # 0:03 to run
def test_training_job_has_ended(out_dir):
    tf.reset_default_graph()
    reset_collections()
    subprocess.check_call(
        [
            sys.executable,
            "examples/tensorflow/scripts/simple.py",
            "--smdebug_path",
            out_dir,
            "--steps",
            "10",
            "--save_frequency",
            "5",
        ],
        env={"CUDA_VISIBLE_DEVICES": "-1", "SMDEBUG_LOG_LEVEL": "debug"},
    )
    assert has_training_ended(out_dir) == True
