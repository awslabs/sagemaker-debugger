# Standard Library
import json


def get_json_config(local_path):
    json_config = {
        "S3Path": "s3://kjndjknd_bucket/prefix",
        "LocalPath": local_path,
        "HookParameters": {"save_all": False, "save_steps": "0,1,2,3"},
        "CollectionConfigurations": [
            {
                "CollectionName": "losses",
                "CollectionParameters": {"save_steps": "1,2,3", "start_step": 1},
            }
        ],
    }

    return json.dumps(json_config)


def get_json_config_full(local_path):
    json_config = {
        "S3Path": "s3://kjndjknd_bucket/prefix",
        "LocalPath": local_path,
        "HookParameters": {"save_all": False, "save_steps": "0,1,2,3"},
        "CollectionConfigurations": [
            {
                "CollectionName": "losses",
                "CollectionParameters": {"save_steps": "1,2,3", "start_step": 1},
            }
        ],
    }
    return json.dumps(json_config)
