# Standard Library
import os
import subprocess
import sys


def launch_smdataparallel_job(script_file_path, script_args, num_workers, config_file_path, mode):
    command = ["smddpsinglenode"] + [sys.executable, script_file_path] + script_args
    env_dict = os.environ.copy()
    env_dict["SMDEBUG_CONFIG_FILE_PATH"] = f"{config_file_path}"
    env_dict["PYTHONPATH"] = "/home/ubuntu/sagemaker-debugger/"
    subprocess.check_call(command, env=env_dict)


def is_gpu_available(framework):
    if framework == "tensorflow2":
        import tensorflow as tf

        return tf.config.list_physical_devices("GPU") > 0
    elif framework == "pytorch":
        import torch

        return torch.cuda.is_available()
    else:
        raise Exception("Invalid framework passed in.")
