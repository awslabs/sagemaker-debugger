import json
import os

from tornasole.core.modes import ModeKeys, ALLOWED_MODE_NAMES
from tornasole.core.logger import get_logger
from tornasole import ReductionConfig, SaveConfig

TORNASOLE_CONFIG_DEFAULT_WORKER_NAME = 'worker0'
TORNASOLE_CONFIG_FILE_PATH_ENV_STR = 'TORNASOLE_CONFIG_FILE_PATH'
TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY = "reduction_configs"
TORNASOLE_CONFIG_SAVE_CONFIGS_KEY = "save_configs"
TORNASOLE_CONFIG_COLLECTIONS_CONFIGS_KEY = "collections"
TORNASOLE_CONFIG_OUTDIR_KEY = "local_path"
TORNASOLE_CONFIG_DRYRUN_KEY = "dry_run"
TORNASOLE_CONFIG_RDN_CFG_KEY = "reduction_config"
TORNASOLE_CONFIG_SAVE_CONFIG_KEY = "save_config"
TORNASOLE_CONFIG_INCLUDE_REGEX_KEY = "include_regex"
TORNASOLE_CONFIG_INCLUDE_COLLECTION_KEY = "include_collections"
TORNASOLE_CONFIG_SAVE_ALL_KEY = "save_all"
DEFAULT_SAGEMAKER_TORNASOLE_PATH = "/opt/ml/output/tensors"
TORNASOLE_CONFIG_INCLUDED_PARAMS = {TORNASOLE_CONFIG_OUTDIR_KEY, 
                                    TORNASOLE_CONFIG_RDN_CFG_KEY,
                                    TORNASOLE_CONFIG_SAVE_CONFIG_KEY, 
                                    TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY,
                                    TORNASOLE_CONFIG_SAVE_CONFIGS_KEY,
                                    TORNASOLE_CONFIG_COLLECTIONS_CONFIGS_KEY}


def _get_save_config_for_name(tornasole_params_dict, save_config_name):
    if save_config_name in tornasole_params_dict[TORNASOLE_CONFIG_SAVE_CONFIGS_KEY]:
        return tornasole_params_dict[TORNASOLE_CONFIG_SAVE_CONFIGS_KEY][save_config_name]
    else:
        raise ValueError("There is no save config defined in json "
                         "element save_configs with "
                         "save_config_name {}".format(save_config_name))


def _get_redn_config_for_name(tornasole_params_dict, redn_config_name):
    if redn_config_name in tornasole_params_dict[TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY]:
        return tornasole_params_dict[TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY][redn_config_name]
    else:
        raise ValueError("There is no reduction config defined in "
                         "json element reduction_configs "
                         "with redn_config_name {}".format(redn_config_name))


def _get_mode_save_configs(save_config_val, tornasole_params_dict):
    if isinstance(save_config_val, dict):
        sc_dict = {}
        for k, v in save_config_val.items():
            if k not in ALLOWED_MODE_NAMES:
                raise ValueError('Invalid key {} for the dictionary '
                                 'mapped to save_config. Valid mode keys are {}.'
                                 .format(k, ','.join(ALLOWED_MODE_NAMES)))
            else:
                sc_dict[ModeKeys[k]] = _get_save_config_for_name(tornasole_params_dict, v)
        return sc_dict
    else:
        return _get_save_config_for_name(tornasole_params_dict, save_config_val)


def collect_tornasole_config_params(collection_manager):
    #default path is sagemaker path
    json_config_file_path = os.getenv(TORNASOLE_CONFIG_FILE_PATH_ENV_STR,
                                      '/opt/ml/input/data/tornasole-config/tornasole-hook-config.json')
    tornasole_params_dict = {}
    params_dict = {}
    if os.path.exists(json_config_file_path):
        with open(json_config_file_path) as json_config_file :
            params_dict = json.load(json_config_file)
        # create reduction config params
        tornasole_params_dict[TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY] = {}
        if TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY in params_dict:
            for redn_configs in params_dict[TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY]:
                for redn_name, redn_json in redn_configs.items():
                    tornasole_params_dict[TORNASOLE_CONFIG_REDUCTION_CONFIGS_KEY][redn_name] = ReductionConfig.from_json(redn_json)

        # create save config params
        tornasole_params_dict[TORNASOLE_CONFIG_SAVE_CONFIGS_KEY] = {}
        if TORNASOLE_CONFIG_SAVE_CONFIGS_KEY in params_dict:
            for save_cfgs in params_dict[TORNASOLE_CONFIG_SAVE_CONFIGS_KEY]:
                for save_cfg_name, save_cfg_json in save_cfgs.items():
                    tornasole_params_dict[TORNASOLE_CONFIG_SAVE_CONFIGS_KEY][save_cfg_name] = SaveConfig.from_json(save_cfg_json)

        # create collections params
        tornasole_params_dict[TORNASOLE_CONFIG_COLLECTIONS_CONFIGS_KEY] = {}
        if TORNASOLE_CONFIG_COLLECTIONS_CONFIGS_KEY in params_dict:
            for collection_cfg in params_dict[TORNASOLE_CONFIG_COLLECTIONS_CONFIGS_KEY]:
                for coll_cfg_name, coll_cfg_json in collection_cfg.items():
                    include_regex = []
                    sc_value = None
                    reduction_config = None

                    if "include_regex" in coll_cfg_json:
                        include_regex = coll_cfg_json['include_regex']
                    if "save_config" in coll_cfg_json:
                        sc_value = _get_mode_save_configs(coll_cfg_json['save_config'],
                                                          tornasole_params_dict)

                    if "reduction_config" in coll_cfg_json:
                        reduction_config = _get_redn_config_for_name(tornasole_params_dict,
                                                                     coll_cfg_json['reduction_config'])

                    collection_manager.add(coll_cfg_name)
                    coll_object = collection_manager.get(coll_cfg_name)
                    coll_object.set_save_config(sc_value)
                    coll_object.set_reduction_config(reduction_config)
                    coll_object.include(include_regex)
                    tornasole_params_dict[TORNASOLE_CONFIG_COLLECTIONS_CONFIGS_KEY][coll_cfg_name] = coll_object

    else:
        get_logger().info("json config file path {} doesn't exist. "
                          "Creating a default hook.".format(json_config_file_path))
    # check local path
    if TORNASOLE_CONFIG_OUTDIR_KEY in params_dict:
        tornasole_params_dict["out_dir"] = params_dict[TORNASOLE_CONFIG_OUTDIR_KEY]
    else:
        ### TODO this is hardcoding done because of Sagemaker, they dont have way to specify this for now
        tornasole_params_dict["out_dir"] = DEFAULT_SAGEMAKER_TORNASOLE_PATH

    if TORNASOLE_CONFIG_RDN_CFG_KEY in params_dict:
        reduction_config_name = params_dict[TORNASOLE_CONFIG_RDN_CFG_KEY]
        tornasole_params_dict[TORNASOLE_CONFIG_RDN_CFG_KEY] = _get_redn_config_for_name(tornasole_params_dict, reduction_config_name)
    else:
        tornasole_params_dict[TORNASOLE_CONFIG_RDN_CFG_KEY] = None

    if TORNASOLE_CONFIG_SAVE_CONFIG_KEY in params_dict:
        sc_value = _get_mode_save_configs(params_dict[TORNASOLE_CONFIG_SAVE_CONFIG_KEY], tornasole_params_dict)
        tornasole_params_dict[TORNASOLE_CONFIG_SAVE_CONFIG_KEY] = sc_value
    else:
        tornasole_params_dict[TORNASOLE_CONFIG_SAVE_CONFIG_KEY] = SaveConfig()
    if TORNASOLE_CONFIG_INCLUDE_REGEX_KEY not in params_dict:
        tornasole_params_dict[TORNASOLE_CONFIG_INCLUDE_REGEX_KEY] = None

    for param in params_dict:
        if param not in TORNASOLE_CONFIG_INCLUDED_PARAMS:
            tornasole_params_dict[param] = params_dict[param]
    return tornasole_params_dict


def create_hook_from_json_config(cls, collection_manager, default_include_collections):
    tornasole_params =  collect_tornasole_config_params(collection_manager)
    for obj in tornasole_params.get(TORNASOLE_CONFIG_COLLECTIONS_CONFIGS_KEY,{}).values():
        collection_manager.add(obj)
    out_dir = tornasole_params.get("out_dir", DEFAULT_SAGEMAKER_TORNASOLE_PATH)
    dry_run = tornasole_params.get("dry_run", False)
    worker = tornasole_params.get("worker", TORNASOLE_CONFIG_DEFAULT_WORKER_NAME)
    reduction_config = tornasole_params.get(TORNASOLE_CONFIG_RDN_CFG_KEY, None)
    save_config = tornasole_params.get(TORNASOLE_CONFIG_SAVE_CONFIG_KEY, SaveConfig())
    include_regex = tornasole_params.get(TORNASOLE_CONFIG_INCLUDE_REGEX_KEY, None)
    include_collections = tornasole_params.get(TORNASOLE_CONFIG_INCLUDE_COLLECTION_KEY,
                                               default_include_collections)
    save_all = tornasole_params.get(TORNASOLE_CONFIG_SAVE_ALL_KEY, False)
    return cls(out_dir, dry_run, worker, reduction_config, save_config,
               include_regex, include_collections, save_all)