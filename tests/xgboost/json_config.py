# Standard Library
import json


def get_json_config(local_path):
    json_config = {
        "S3Path": "s3://kjndjknd_bucket/prefix",
        "LocalPath": local_path,
        "HookParameters": {"save_all": False},
        "CollectionConfigurations": [
            {"CollectionName": "losses", "CollectionParameters": {"save_interval": 1}}
        ],
    }

    return json.dumps(json_config)


def get_json_config_full(local_path):
    json_config = {
        "S3Path": "s3://kjndjknd_bucket/prefix",
        "LocalPath": local_path,
        "HookParameters": {"save_all": False},
        "CollectionConfigurations": [
            {"CollectionName": "losses", "CollectionParameters": {"save_interval": 1}}
        ],
    }
    return json.dumps(json_config)
