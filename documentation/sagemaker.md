# SageMaker Examples

There are two cases for using SageMaker: fully managed or bring-your-own-container (BYOC).
In fully managed mode, SageMaker will automatically inject hooks into your training script - no code
change necessary! This is supported for TensorFlow 1.15, PyTorch 1.3, and MXNet 1.6.

In BYOC mode, you will need to instantiate the hook and use it yourself. Built-in rules will not be
available, but you can write custom rules and use those.

## Example Usage (Sagemaker Fully Managed)
This setup will work for any script without code changes. This exa Tensorflow 1.15.
See the AWS docs for greater details on the JSON configuration.

This example uses TensorFlow.
To use PyTorch or MXNet, simply call `sagemaker.pytorch.PyTorch` or `sagemaker.mxnet.MXNet`.
```
import sagemaker
from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig

hook_config = DebuggerHookConfig(
    s3_output_path = args.s3_path,
    container_local_path = args.local_path,
    hook_parameters = {
        "save_steps": "0,20,40,60,80"
    },
    collection_configs = {
        { "CollectionName": "weights" },
        { "CollectionName": "biases" },
    },
)


rule = Rule.sagemaker(
    rule_configs.exploding_tensor(),
    rule_parameters={
        "tensor_regex": ".*"
    },
    collections_to_save=[
        CollectionConfig(name="weights", parameters={}),
        CollectionConfig(name="losses", parameters={}),
    ],
)

sagemaker_simple_estimator = sagemaker.tensorflow.TensorFlow(
    entry_point=simple_entry_point_script,
    role=sagemaker.get_execution_role(),
    base_job_name=args.job_name,
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="1.15",
    py_version="py3",
    debugger_hook_config=hook_config,
    rules=[rule],
)

sagemaker_simple_estimator.fit()
```

When a rule triggers, it will create a CloudWatch event.

## Example Usage (SageMaker BYOC)
Use the same script as fully managed. In the script, call
`hook = smd.{hook_class}.create_from_json_file()`
to get the hook and then use it as described in the rest of the API docs.
