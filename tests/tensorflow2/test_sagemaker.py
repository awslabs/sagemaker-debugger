# First Party
# Third Party
from tests.utils import SagemakerSimulator

import smdebug.tensorflow as smd


def test_sagemaker():
    json_file_contents = """
{
    "S3OutputPath": "s3://sagemaker-test",
    "LocalPath": "/opt/ml/output/tensors",
    "HookParameters": null,
    "CollectionConfigurations": [
        {
            "CollectionName": "weights",
            "CollectionParameters": null
        },
        {
            "CollectionName": "losses",
            "CollectionParameters": null
        }
    ],
    "DebugHookSpecification": null
}
"""
    with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
        smd.del_hook()
        hook = smd.get_hook(hook_type="keras", create_if_not_exists=True)
        print(hook)
        assert "weights" in hook.include_collections, hook
