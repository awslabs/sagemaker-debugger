## Example Usage (Sagemaker)
This setup will work for any script without code changes. Note that you must use Tensorflow 1.15.
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
    rules=[rule],
)

sagemaker_simple_estimator.fit(wait=False)
```

When a rule triggers, it will create a CloudWatch event.
