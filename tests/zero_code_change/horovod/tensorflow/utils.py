# Standard Library
import json
import subprocess
import sys
from pathlib import Path

# Third Party
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


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
    with open(path.absolute(), "w") as outfile:
        json.dump(config_dict, outfile)
    return path.absolute()


def launch_horovod_job(script_file_path, script_args, num_workers, config_file_path, mode):
    subprocess.check_call(
        ["horovodrun", "-np", str(num_workers)]
        + ([] if mode == "gpu" else ["-x", "CUDA_VISIBLE_DEVICES=-1"])
        + [sys.executable, script_file_path]
        + script_args,
        env={
            "SMDEBUG_CONFIG_FILE_PATH": config_file_path,
            "PYTHONPATH": "/home/ubuntu/sagemaker-debugger/",
        },
    )
