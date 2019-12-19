"""
Example JSON config:

{
  "S3Path": "s3://bucket/prefix",
  "LocalPath": "/tmp/test_hook_from_json_config_full",
  "HookParameters": {
    "export_tensorboard": true,
    "tensorboard_dir": "/tmp/tensorboard",
    "save_all": false,
    "include_regex": "regexe1,regex2",
    "save_interval": 100,
    "save_steps": "1,2,3,4",
    "start_step": 1,
    "reductions": "min,max,mean,std,abs_variance,abs_sum,abs_l2_norm"
  },
  "CollectionConfigurations": [
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

# Standard Library
import json
import os
from pathlib import Path
from typing import Dict, Optional

# First Party
from smdebug import ReductionConfig, SaveConfig, SaveConfigMode
from smdebug.analysis.utils import parse_bool
from smdebug.core.config_constants import (
    CONFIG_COLLECTION_CONFIG_KEY,
    CONFIG_COLLECTION_NAME_KEY,
    CONFIG_COLLECTION_PARAMS_KEY,
    CONFIG_FILE_PATH_ENV_STR,
    CONFIG_HOOK_PARAMS_KEY,
    CONFIG_INCLUDE_REGEX_KEY,
    CONFIG_INCLUDE_WORKERS_KEY,
    CONFIG_OUTDIR_KEY,
    CONFIG_RDN_CFG_KEY,
    CONFIG_REDUCTION_CONFIGS_KEY,
    CONFIG_SAVE_ALL_KEY,
    CONFIG_SAVE_CONFIGS_KEY,
    DEFAULT_CONFIG_FILE_PATH,
    DEFAULT_SAGEMAKER_OUTDIR,
    DEFAULT_SAGEMAKER_TENSORBOARD_PATH,
    DEFAULT_WORKER_NAME,
    EXPORT_TENSORBOARD_KEY,
    TENSORBOARD_CONFIG_FILE_PATH_ENV_STR,
    TENSORBOARD_DIR_KEY,
)
from smdebug.core.logger import get_logger
from smdebug.core.modes import ModeKeys
from smdebug.core.sagemaker_utils import is_sagemaker_job
from smdebug.core.utils import merge_two_dicts, split


def get_json_config_as_dict(json_config_path) -> Dict:
    """Checks json_config_path, then environment variables, then attempts to load.

    Will throw FileNotFoundError if a config is not available.
    """
    if json_config_path is not None:
        path = json_config_path
    else:
        path = os.getenv(CONFIG_FILE_PATH_ENV_STR, DEFAULT_CONFIG_FILE_PATH)
    with open(path) as json_config_file:
        params_dict = json.load(json_config_file)
    get_logger().info(f"Creating hook from json_config at {path}.")
    return params_dict


def get_tensorboard_dir_from_json_config() -> Optional[str]:
    """ Expects tb_json_config_path to contain { “LocalPath”: /my/tensorboard/path }.

    Returns the path contained in that json file, or None if not exists.
    """
    tb_json_config_path = os.getenv(
        TENSORBOARD_CONFIG_FILE_PATH_ENV_STR, DEFAULT_SAGEMAKER_TENSORBOARD_PATH
    )
    path = Path(tb_json_config_path)
    if path.is_file():
        my_dict = json.loads(path.read_text())
        tensorboard_out_dir = my_dict.get("LocalPath")
        return tensorboard_out_dir
    else:
        return None


def collect_hook_config_params(params_dict) -> Dict:
    """Read the config file from an environment variable and return a dictionary.

    Return a dictionary, example keys:
    dict_keys(['reduction_configs', 'save_configs', 'collections', 'out_dir', 'reduction_config', 'save_config',
    'include_regex', 'config_name', 's3_path'])
    """
    # Build params dictionary from the json file

    # Declare defaults
    parsed_params_dict = {
        CONFIG_RDN_CFG_KEY: None,
        CONFIG_REDUCTION_CONFIGS_KEY: {},
        CONFIG_SAVE_CONFIGS_KEY: {},
        CONFIG_INCLUDE_REGEX_KEY: None,
    }
    # Set top-level path parameters
    # SageMaker doesn't have any way to specify this for now, so default to using their path
    parsed_params_dict["out_dir"] = params_dict.get(CONFIG_OUTDIR_KEY, DEFAULT_SAGEMAKER_OUTDIR)

    # Get the main HookParameters; pass these as defaults
    hook_params = params_dict.get(CONFIG_HOOK_PARAMS_KEY, {})
    # If we have {"HookParameters": null}, replace null with {}.
    hook_params = {} if hook_params is None else hook_params
    base_config_modes = parse_save_config_modes_dict(params=hook_params)
    parsed_params_dict["save_config_modes"] = base_config_modes
    # If we pass reduction=None, then the full tensor is saved by default
    if "reductions" in hook_params:
        parsed_params_dict[CONFIG_RDN_CFG_KEY] = ReductionConfig.from_dict(hook_params)
    if "save_all" in hook_params:
        parsed_params_dict[CONFIG_SAVE_ALL_KEY] = parse_bool(hook_params["save_all"], False)
    if "include_regex" in hook_params:
        parsed_params_dict[CONFIG_INCLUDE_REGEX_KEY] = split(hook_params["include_regex"])
    if CONFIG_INCLUDE_WORKERS_KEY in hook_params:
        parsed_params_dict[CONFIG_INCLUDE_WORKERS_KEY] = hook_params[CONFIG_INCLUDE_WORKERS_KEY]
    parsed_params_dict[EXPORT_TENSORBOARD_KEY] = parse_bool(
        hook_params.get(EXPORT_TENSORBOARD_KEY, False), False
    )
    parsed_params_dict[TENSORBOARD_DIR_KEY] = hook_params.get(TENSORBOARD_DIR_KEY, None)
    return parsed_params_dict


def get_include_collections(params_dict):
    if (
        CONFIG_COLLECTION_CONFIG_KEY in params_dict
        and params_dict[CONFIG_COLLECTION_CONFIG_KEY] is not None
    ):
        include_collections = []
        for config in params_dict[CONFIG_COLLECTION_CONFIG_KEY]:
            # Require name and parameters for each collection.
            if CONFIG_COLLECTION_NAME_KEY not in config:
                raise ValueError(f"Must specify '{CONFIG_COLLECTION_NAME_KEY}' in JSON config.")
            include_collections.append(config[CONFIG_COLLECTION_NAME_KEY])
    else:
        include_collections = None
    return include_collections


def add_collections_to_manager(collection_manager, params_dict, hook_params):
    """Read the config file from an environment variable and return a dictionary.

    Return a dictionary, example keys:
    dict_keys(['reduction_configs', 'save_configs', 'collections', 'out_dir', 'reduction_config', 'save_config',
    'include_regex', 'config_name', 's3_path'])
    """
    # For each collection configuration, also create the reduction config and save config.
    if (
        CONFIG_COLLECTION_CONFIG_KEY in params_dict
        and params_dict[CONFIG_COLLECTION_CONFIG_KEY] is not None
    ):
        for config in params_dict[CONFIG_COLLECTION_CONFIG_KEY]:
            # Require name and parameters for each collection.
            if CONFIG_COLLECTION_NAME_KEY not in config:
                raise ValueError(f"Must specify '{CONFIG_COLLECTION_NAME_KEY}' in JSON config.")

            name = config[CONFIG_COLLECTION_NAME_KEY]
            coll_params = config.get(CONFIG_COLLECTION_PARAMS_KEY, {})
            # If we have {"CollectionParameters": null}, replace null with {}.
            coll_params = {} if coll_params is None else coll_params
            coll = collection_manager.get(name)
            coll_config_modes = parse_save_config_modes_dict(
                params=coll_params, base_config_modes=hook_params["save_config_modes"]
            )
            mode_save_configs = {
                mode: SaveConfigMode.from_dict(val) for mode, val in coll_config_modes.items()
            }
            coll.save_config = mode_save_configs
            if "reductions" in coll_params:
                coll.reduction_config = ReductionConfig.from_dict(coll_params)
            if "include_regex" in coll_params:
                coll.include(split(coll_params["include_regex"]))
            if "save_histogram" in coll_params:
                coll.save_histogram = parse_bool(coll_params["save_histogram"], True)


def create_hook_from_json_config(hook_cls, json_config_path, default_values=None):
    """Returns a SessionHook object corresponding to either TF, PT, or MXNet.

    If json_config_path is None, an environment variable must be set.
    Here we compare HookParameters with CollectionConfiguration and set all the defaults.
    """
    params_dict = get_json_config_as_dict(json_config_path=json_config_path)
    hook_params = collect_hook_config_params(params_dict)

    out_dir = hook_params.get("out_dir")
    dry_run = hook_params.get("dry_run", False)
    reduction_config = hook_params.get(CONFIG_RDN_CFG_KEY, None)
    save_config = SaveConfig.from_dict(hook_params.get("save_config_modes"), default_values)
    include_regex = hook_params.get(CONFIG_INCLUDE_REGEX_KEY)
    include_collections = get_include_collections(params_dict)
    save_all = hook_params.get(CONFIG_SAVE_ALL_KEY, False)
    include_workers = hook_params.get(CONFIG_INCLUDE_WORKERS_KEY, "one")

    # If Sagemaker, emit TB only if JSON file exists
    if is_sagemaker_job():
        tensorboard_dir = get_tensorboard_dir_from_json_config()
        export_tensorboard = bool(tensorboard_dir is not None)
    # Otherwise, place TB artifacts in out_dir
    else:
        tensorboard_dir = hook_params[TENSORBOARD_DIR_KEY]
        export_tensorboard = hook_params[EXPORT_TENSORBOARD_KEY]

    hook = hook_cls(
        out_dir=out_dir,
        export_tensorboard=export_tensorboard,
        tensorboard_dir=tensorboard_dir,
        dry_run=dry_run,
        reduction_config=reduction_config,
        save_config=save_config,
        include_regex=include_regex,
        include_collections=include_collections,
        include_workers=include_workers,
        save_all=save_all,
    )
    add_collections_to_manager(hook.collection_manager, params_dict, hook_params)
    return hook


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
