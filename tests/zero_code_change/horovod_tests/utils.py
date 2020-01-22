# Standard Library
import json
import os
import subprocess
import sys
from pathlib import Path


def build_json(out_dir, include_workers="all", include_collections=None, path=None, save_all=False):
    if include_collections is None:
        include_collections = ["weights", "gradients"]
    if path is None:
        path = Path(out_dir).joinpath("config.json")
    config_dict = {}
    config_dict["LocalPath"] = out_dir
    config_dict["HookParameters"] = {"include_workers": include_workers, "save_all": save_all}
    config_dict["CollectionConfigurations"] = []
    for ic in include_collections:
        config_dict["CollectionConfigurations"].append({"CollectionName": ic})
    os.makedirs(out_dir, exist_ok=True)
    with open(path.absolute(), "w") as outfile:
        json.dump(config_dict, outfile)
    return path.absolute()


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
