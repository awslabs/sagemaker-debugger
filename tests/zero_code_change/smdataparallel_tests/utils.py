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
