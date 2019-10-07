"""
Example JSON config:

{
  "S3Path": "s3://bucket/prefix",
  "LocalPath": "newlogsRunTest/test_hook_from_json_config_full",
  "HookParameters": {
    "save_all": false,
    "include_regex": "regexe1,regex2",
    "save_interval": 100,
    "save_steps": "1,2,3,4",
    "start_step": 1,
    "reductions": "min,max,mean,std,abs_variance,abs_sum,abs_l2_norm"
  },
  "CollectionConfiguration": [
    {
      "CollectionName": "collection_obj_name1",
      "CollectionParameters": {
        "include_regex": "regexe5*",
        "save_interval": 100,
        "save_steps": "1,2,3",
        "start_step": 1,
        "reductions": "min,abs_max,l1_norm,abs_l2_norm",
      }
    },
    {
      "CollectionName": "collection_obj_name2",
      "CollectionParameters": {
        "include_regex": "regexe6*",
        "train.save_interval": 100,
        "eval.save_interval": 1,
        "save_steps": "1,2,3",
        "start_step": 1,
        "reductions": "min,abs_max,l1_norm,abs_l2_norm"
      }
    }
  ]
}
"""

import json
import os
from typing import Dict

from tornasole.core.modes import ModeKeys
from tornasole.core.logger import get_logger
from tornasole.core.utils import merge_two_dicts, split
from tornasole import ReductionConfig, SaveConfig, SaveConfigMode

from tornasole.core.config_constants import TORNASOLE_CONFIG_DEFAULT_WORKER_NAME, TORNASOLE_CONFIG_FILE_PATH_ENV_STR, \
    DEFAULT_CONFIG_FILE_PATH, TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY, TORNASOLE_CONFIG_SAVE_CONFIGS_KEY, \
    TORNASOLE_CONFIG_OUTDIR_KEY, TORNASOLE_CONFIG_RDN_CFG_KEY, TORNASOLE_CONFIG_INCLUDE_REGEX_KEY, \
    TORNASOLE_CONFIG_SAVE_ALL_KEY, DEFAULT_SAGEMAKER_TORNASOLE_PATH


def create_hook_from_json_config(hook_cls, collection_manager):
    """Returns a TornasoleHook object corresponding to either TF, PT, or MXNet.

    Here we compare HookParameters with CollectionConfiguration and set all the defaults.
    """
    tornasole_params = collect_tornasole_config_params(collection_manager)
    if "collections" in tornasole_params:
        include_collections = []
        for obj in tornasole_params["collections"].values():
            include_collections.append(obj.name)
            collection_manager.add(obj)
    else:
        include_collections = None

    out_dir = tornasole_params.get("out_dir", DEFAULT_SAGEMAKER_TORNASOLE_PATH)
    dry_run = tornasole_params.get("dry_run", False)
    reduction_config = tornasole_params.get(TORNASOLE_CONFIG_RDN_CFG_KEY)
    save_config = SaveConfig.from_dict(tornasole_params.get("save_config_modes"))
    include_regex = tornasole_params.get(TORNASOLE_CONFIG_INCLUDE_REGEX_KEY)
    save_all = tornasole_params.get(TORNASOLE_CONFIG_SAVE_ALL_KEY, False)
    return hook_cls(
        out_dir,
        dry_run,
        reduction_config,
        save_config,
        include_regex,
        include_collections,
        save_all,
    )


def collect_tornasole_config_params(collection_manager) -> Dict:
    """Read the config file from an environment variable and return a dictionary.

    Return a dictionary, example keys:
    dict_keys(['reduction_configs', 'save_configs', 'collections', 'out_dir', 'reduction_config', 'save_config',
    'include_regex', 'config_name', 's3_path'])
    """
    # Build params dictionary if given a json file, otherwise leave it empty
    params_dict = {}
    json_config_file_path = os.getenv(TORNASOLE_CONFIG_FILE_PATH_ENV_STR, DEFAULT_CONFIG_FILE_PATH)
    if os.path.exists(json_config_file_path):
        with open(json_config_file_path) as json_config_file:
            params_dict = json.load(json_config_file)
    else:
        get_logger().info(
            f"json config file path {json_config_file_path} doesn't exist. Creating a default hook."
        )

    # Declare defaults
    tornasole_params_dict = {
        TORNASOLE_CONFIG_RDN_CFG_KEY: None,
        TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY: {},
        TORNASOLE_CONFIG_SAVE_CONFIGS_KEY: {},
        TORNASOLE_CONFIG_INCLUDE_REGEX_KEY: None,
    }
    # Set top-level path parameters
    # SageMaker doesn't have any way to specify this for now, so default to using their path
    tornasole_params_dict["out_dir"] = params_dict.get(
        TORNASOLE_CONFIG_OUTDIR_KEY, DEFAULT_SAGEMAKER_TORNASOLE_PATH
    )

    # Get the main HookParameters; pass these as defaults
    hook_params = params_dict.get("HookParameters", {})
    base_config_modes = parse_save_config_modes_dict(params=hook_params)
    tornasole_params_dict["save_config_modes"] = base_config_modes

    # If we pass reduction=None, then the full tensor is saved by default
    if "reductions" in hook_params:
        tornasole_params_dict[TORNASOLE_CONFIG_RDN_CFG_KEY] = ReductionConfig.from_dict(hook_params)
    if "save_all" in hook_params:
        tornasole_params_dict[TORNASOLE_CONFIG_SAVE_ALL_KEY] = hook_params["save_all"]
    if "include_regex" in hook_params:
        tornasole_params_dict[TORNASOLE_CONFIG_INCLUDE_REGEX_KEY] = split(
            hook_params["include_regex"]
        )

    # For each collection configuration, also create the reduction config and save config.
    if "CollectionConfiguration" in params_dict:
        tornasole_params_dict["collections"] = {}
        for config in params_dict["CollectionConfiguration"]:
            # Require name and parameters for each collection.
            if "CollectionName" not in config:
                raise ValueError("Must specify 'CollectionName' in JSON config.")

            name = config["CollectionName"]
            coll_params = config.get("CollectionParameters", {})

            collection_manager.add(name)
            coll = collection_manager.get(name)
            coll_config_modes = parse_save_config_modes_dict(
                params=coll_params, base_config_modes=base_config_modes
            )
            mode_save_configs = {
                mode: SaveConfigMode.from_dict(val)
                for mode, val in coll_config_modes.items()
            }
            coll.set_save_config(mode_save_configs)
            if "reductions" in coll_params:
                coll.set_reduction_config(ReductionConfig.from_dict(coll_params))
            if "include_regex" in coll_params:
                coll.include(split(coll_params["include_regex"]))
            tornasole_params_dict["collections"][name] = coll

    return tornasole_params_dict


def parse_save_config_modes_dict(params, base_config_modes=None) -> Dict:
    """Create a nested dict (ModeKeys -> settings) from a flattened dict.

    Optionally pass in a base_config_modes dict, mapping ModeKeys -> dict.
    This will be the default value for certain settings if none are specified.
    """
    base_config = parse_save_config_dict(params=params, mode=None)
    # For each mode, start with the base_config and override on mode-specific keys.
    configs = {
        mode: merge_two_dicts(base_config, parse_save_config_dict(params=params, mode=mode))
        for mode in ModeKeys
    }
    # Apply optional defaults
    if base_config_modes:
        for mode in ModeKeys:
            # `configs` takes precedence over `base_config_modes`
            configs[mode] = merge_two_dicts(base_config_modes[mode], configs[mode])
    return configs


def parse_save_config_dict(params, mode=None) -> Dict:
    """Grab the relevant keys for a SaveConfig and return a dictionary."""
    if not isinstance(params, dict):
        raise ValueError("parameter must be dict")

    if mode is None:
        prefix = ""
    elif mode == ModeKeys.TRAIN:
        prefix = "train."
    elif mode == ModeKeys.EVAL:
        prefix = "eval."
    elif mode == ModeKeys.PREDICT:
        prefix = "predict."
    elif mode == ModeKeys.GLOBAL:
        prefix = "global."
    else:
        raise ValueError(f"Invalid mode={mode}.")

    # Only look at keys starting with prefix
    params = {key[len(prefix) :]: value for key, value in params.items() if key.startswith(prefix)}

    # Parse relevant key-value pairs and place them in a new `ret` dictionary
    ret = {}
    if "save_interval" in params:
        ret["save_interval"] = params["save_interval"]
    if "save_steps" in params:
        ret["save_steps"] = [int(x) for x in split(params["save_steps"])]
    if "start_step" in params:
        ret["start_step"] = params["start_step"]
    if "end_step" in params:
        ret["end_step"] = params["end_step"]
    return ret
