# Standard Library
import os
import shutil
from pathlib import Path

# Third Party
import boto3
from tests.constants import TEST_DATASET_S3_PATH

# First Party
from smdebug.core.config_constants import (
    CONFIG_FILE_PATH_ENV_STR,
    DEFAULT_SAGEMAKER_OUTDIR,
    DEFAULT_SAGEMAKER_TENSORBOARD_PATH,
    DEFAULT_WORKER_NAME,
    TENSORBOARD_CONFIG_FILE_PATH_ENV_STR,
)
from smdebug.core.utils import is_s3, remove_file_if_exists
from smdebug.trials import create_trial


def use_s3_datasets():
    s3 = boto3.resource("s3")
    _, bucket, _ = is_s3(TEST_DATASET_S3_PATH)
    try:
        s3.meta.client.head_bucket(Bucket=bucket)
        return True
    except Exception:
        return False


def verify_shapes(out_dir, step_num, num_tensors, exact_equal=True):
    trial = create_trial(out_dir)
    tnames = trial.tensor_names(step=step_num)
    if exact_equal:
        assert num_tensors == len(tnames), (len(tnames), tnames)
    else:
        assert num_tensors >= len(tnames), (len(tnames), tnames)
    for tname in tnames:
        tensor = trial.tensor(tname)
        assert isinstance(tensor.shape(step_num), tuple), (tname, tensor.shape(step_num))


class SagemakerSimulator(object):
    """
    Creates an environment variable pointing to a JSON config file, and creates the config file.
    Used for integration testing with zero-code-change.

    If `disable=True`, then we still create the `out_dir` directory, but ignore the config file.
    """

    def __init__(
        self,
        json_config_path="/tmp/zcc_config.json",
        tensorboard_dir="/tmp/tensorboard",
        training_job_name="sm_job",
        json_file_contents="{}",
        enable_tb=True,
        cleanup=True,
    ):
        self.out_dir = DEFAULT_SAGEMAKER_OUTDIR
        self.json_config_path = json_config_path
        self.tb_json_config_path = DEFAULT_SAGEMAKER_TENSORBOARD_PATH
        self.tensorboard_dir = tensorboard_dir
        self.training_job_name = training_job_name
        self.json_file_contents = json_file_contents
        self.enable_tb = enable_tb
        self.cleanup = cleanup

    def __enter__(self):
        if self.cleanup is True:
            shutil.rmtree(self.out_dir, ignore_errors=True)
        shutil.rmtree(self.json_config_path, ignore_errors=True)
        tb_parent_dir = str(Path(self.tb_json_config_path).parent)
        shutil.rmtree(tb_parent_dir, ignore_errors=True)

        os.environ[CONFIG_FILE_PATH_ENV_STR] = self.json_config_path
        os.environ["TRAINING_JOB_NAME"] = self.training_job_name
        with open(self.json_config_path, "w+") as my_file:
            # We'll just use the defaults, but the file is expected to exist
            my_file.write(self.json_file_contents)

        if self.enable_tb is True:
            os.environ[TENSORBOARD_CONFIG_FILE_PATH_ENV_STR] = self.tb_json_config_path
            os.makedirs(tb_parent_dir, exist_ok=True)
            with open(self.tb_json_config_path, "w+") as my_file:
                my_file.write(
                    f"""
                    {{
                        "LocalPath": "{self.tensorboard_dir}"
                    }}
                    """
                )

        return self

    def __exit__(self, *args):
        # Throws errors when the writers try to close.
        # shutil.rmtree(self.out_dir, ignore_errors=True)
        if self.cleanup is True:
            remove_file_if_exists(self.json_config_path)
            remove_file_if_exists(self.tb_json_config_path)
            if CONFIG_FILE_PATH_ENV_STR in os.environ:
                del os.environ[CONFIG_FILE_PATH_ENV_STR]
            if "TRAINING_JOB_NAME" in os.environ:
                del os.environ["TRAINING_JOB_NAME"]
            if TENSORBOARD_CONFIG_FILE_PATH_ENV_STR in os.environ:
                del os.environ[TENSORBOARD_CONFIG_FILE_PATH_ENV_STR]
