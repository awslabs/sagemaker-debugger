# Standard Library
import json


def get_json_config(local_path):
    json_config = {
        "S3Path": "s3://kjndjknd_bucket/prefix",
        "LocalPath": local_path,
        "HookParameters": {"save_all": False, "save_steps": "0,1,2,3"},
    }

    return json.dumps(json_config)


def get_json_config_full(local_path):
    json_config = {
        "S3Path": "s3://kjndjknd_bucket/prefix",
        "LocalPath": local_path,
        "HookParameters": {
            "include_regex": "regexe1,regex2",
            "reductions": "min,max,mean,std,abs_variance,abs_sum,abs_l2_norm",
            "include_collections": "collection_obj_name1,collection_obj_name2",
            "save_all": False,
            "save_interval": 100,
            "save_steps": "1,2,3,4",
            "start_step": 1,
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
                },
            },
            {
                "CollectionName": "collection_obj_name2",
                "CollectionParameters": {
                    "include_regex": "regexe6*",
                    "save_interval": 100,
                    "save_steps": "1,2,3",
                    "start_step": 1,
                    "reductions": "min,abs_max,l1_norm,abs_l2_norm",
                },
            },
        ],
    }
    return json.dumps(json_config)


def get_json_config_for_losses(local_path):
    json_config = {
        "S3Path": "s3://kjndjknd_bucket/prefix",
        "LocalPath": local_path,
        "HookParameters": {"save_all": False},
        "CollectionConfigurations": [
            {"CollectionName": "losses", "CollectionParameters": {"save_interval": 1}}
        ],
    }
    return json.dumps(json_config)
