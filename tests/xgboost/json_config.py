import json


def get_json_config(local_path):
    json_config = {
        "config_name": "my training tornasole config",
        "s3_path": "s3://kjndjknd_bucket/prefix",
        "local_path": local_path,
        "save_config": "default_save_config",
        "save_all": False,
        "save_configs": [
            {
                "default_save_config": {
                    "save_steps": [0, 1, 2, 3]
                }
            }
        ]
    }

    return json.dumps(json_config)


def get_json_config_full(local_path):
    json_config = {
        "config_name": "my training tornasole config",
        "s3_path": "s3://kjndjknd_bucket/prefix",
        "local_path": local_path,
        "include_regex": ["regexe1", "regex2"],
        "save_config": "name_of_save_Config_json_object",
        "reduction_config": "name_of_redn_config_json_object",
        "include_collections":  [
            "collection_obj_name1", "collection_obj_name2"
        ],
        "save_all": False,
        "collections": [
            {
                "collection_obj_name1": {
                    "include_regex": ["regexe5*"],
                    "save_config": "name_of_save_Config_json__coll_object",
                    "reduction_config": "name_of_redn_config_json_object"
                }
            },
            {
                "collection_obj_name2": {
                    "include_regex": ["regexe6*"],
                    "save_config": "name_of_save_Config_json__coll_object",
                    "reduction_config": "name_of_redn_config_json_object_col"
                }
            }
        ],
        "save_configs": [
            {
                "name_of_save_Config_json_object": {
                    "save_interval": 100,
                    "save_steps": [1, 2, 3, 4],
                    "skip_num_steps": 1,
                    "when_nan": ["tensor1*", "tensor2*"]
                }
            },
            {
                "name_of_save_Config_json__coll_object": {
                    "save_interval": 100,
                    "save_steps": [1, 2, 3],
                    "skip_num_steps": 1,
                    "when_nan": ["tensor3*", "tensor4*"]
                }
            }
        ],
        "reduction_configs": [
            {
                "name_of_redn_config_json_object": {
                    "reductions": ["min", "max", "mean", "std"],
                    "abs_reductions": ["variance", "sum"],
                    "norms": [],
                    "abs_norms": ["l2"]
                }
            },
            {
                "name_of_redn_config_json_object_col": {
                    "reductions": ["min"],
                    "abs_reductions": ["max"],
                    "norms": ["l1"],
                    "abs_norms": ["l2"]
                }
            }
        ]
    }
    return json.dumps(json_config)
