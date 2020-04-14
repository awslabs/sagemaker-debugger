# Standard Library
import os
import subprocess
import sys


def launch_horovod_job(script_file_path, script_args, num_workers, config_file_path, mode):
    command = ["mpirun", "-np", str(num_workers)] + [sys.executable, script_file_path] + script_args
    env_dict = os.environ.copy()
    env_dict["SMDEBUG_CONFIG_FILE_PATH"] = f"{config_file_path}"
    env_dict["PYTHONPATH"] = "/home/ubuntu/sagemaker-debugger/"
    if mode == "cpu":
        env_dict["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.check_call(command, env=env_dict)
