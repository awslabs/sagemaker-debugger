# Future
from __future__ import print_function

# Standard Library
import json
import os
from pathlib import Path


def build_json(
    out_dir,
    include_workers="all",
    include_collections=None,
    path=None,
    save_all=False,
    save_interval=None,
):
    """
    :param out_dir: str
        represents a path into which outputs will be written to
    :param include_workers: str
        makes the hook save data from all workers
    :param include_collections: list of str representing collection names
            takes as input the collections which should be saved.
            if this is empty, it defaults to including all collections from code

    :param path: str
        path at which config.json is found.
    :param save_all:  bool
            a shortcut for saving all tensors in the model.
            they are all saved in the collection `all`
    :param save_interval: int
            step interval at which the hook will save tensors
    :return: str absolute path of the json generated
    """
    if include_collections is None:
        include_collections = ["weights", "gradients"]
    if path is None:
        path = Path(out_dir).joinpath("config.json")
    if save_interval is None:
        save_interval = "500"

    config_dict = {}
    config_dict["LocalPath"] = out_dir
    config_dict["HookParameters"] = {
        "include_workers": include_workers,
        "save_all": save_all,
        "save_interval": save_interval,
    }
    config_dict["CollectionConfigurations"] = []
    for ic in include_collections:
        config_dict["CollectionConfigurations"].append({"CollectionName": ic})
    os.makedirs(out_dir, exist_ok=True)
    with open(path.absolute(), "w") as outfile:
        json.dump(config_dict, outfile)
    return path.absolute()
