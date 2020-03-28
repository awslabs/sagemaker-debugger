# Standard Library
import os
import subprocess
import sys

# Third Party
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def launch_horovod_job(script_file_path, script_args, num_workers, config_file_path, mode):
    command = (
        ["horovodrun", "-np", str(num_workers)] + [sys.executable, script_file_path] + script_args
    )
    env_dict = os.environ.copy()
    env_dict["SMDEBUG_CONFIG_FILE_PATH"] = f"{config_file_path}"
    env_dict["PYTHONPATH"] = "/home/ubuntu/sagemaker-debugger/"
    if mode == "cpu":
        env_dict["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.check_call(command, env=env_dict)
