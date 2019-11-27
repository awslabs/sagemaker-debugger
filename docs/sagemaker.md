# SageMaker

There are two cases for SageMaker:
- Zero-Script-Change (ZSC): Here you specify which rules to use, and run your existing script.
- Bring-Your-Own-Container (BYOC): Here you specify the rules to use, and modify your training script.

Table of Contents
- [Version Support](#version-support)
- [Zero-Script-Change Example](#byoc-example)
- [Bring-Your-Own-Container Example](#byoc-example)

## Configuration Details



```python
rule = sagemaker.debugger.Rule.sagemaker(
    base_config: dict, # Use an import, e.g. sagemaker.debugger.rule_configs.exploding_tensor()
    name: str=None,
    instance_type: str=None,
    container_local_path: str=None,
    volume_size_in_gb: int=None,
    other_trials_s3_input_paths: str=None,
    rule_parameters: dict=None,
    collections_to_save: list[sagemaker.debugger.CollectionConfig]=None,
)
```

```python
hook_config = sagemaker.debugger.DebuggerHookConfig(
    s3_output_path: str,
    container_local_path: str=None,
    hook_parameters: dict=None,
    collection_configs: list[sagemaker.debugger.CollectionConfig]=None,
)
```

```python
tb_config = sagemaker.debugger.TensorBoardOutputConfig(
    s3_output_path: str,
    container_local_path: str=None,
)
```

```python
collection_config = sagemaker.debugger.CollectionConfig(
    name: str,
    parameters: dict,
)
```



## SageMaker Estimator Parameters
There are three parameters to pass into SageMaker Estimator:


## Example Usage (Sagemaker Fully Managed)
This setup will work for any script without code changes. This example shows Tensorflow 1.15.
See the [JSON specification](https://link.com) section of API.md for details on the JSON configuration.

This example uses TensorFlow.
To use PyTorch or MXNet, simply call `sagemaker.pytorch.PyTorch` or `sagemaker.mxnet.MXNet`.
```python
import sagemaker
from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig

hook_config = DebuggerHookConfig(
    s3_output_path = "s3://my-bucket/debugger-logs",
    hook_parameters = {
        "save_steps": "0,20,40,60,80"
    },
    collection_configs = [
        CollectionConfig(name="weights"),
        CollectionConfig(name="biases"),
    ],
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
Define the
Use the same script as fully managed. In the script, call
`hook = smd.{hook_class}.create_from_json_file()`
to get the hook and then use it as described in the rest of the API docs.


## Version Support
In ZSC mode, SageMaker will use custom framework forks to automatically save tensors. This is supported
only for certain Deep Learning Containers.
| DLC Framework | Version |
|---|---|
| TensorFlow | 1.15 |
| PyTorch | 1.3 |
| MXNet | 1.6 |

In BYOC mode, custom framework forks are not available. You must modify your script to save tensors.
This is supported for
| Framework | Versions |
|---|---|
| TensorFlow | 1.13, 1.14, 1.15 |
| PyTorch | 1.2, 1.3 |
| MXNet | 1.4, 1.5, 1.6


## Comprehensive Rule List
Full list of rules is:
| Rule Name | Behavior |
| --- | --- |
| `vanishing_gradient` | Detects a vanishing gradient. |
| `all_zero` | ??? |
| `check_input_images` | ??? |
| `similar_across_runs` | ??? |
| `weight_update_ratio` | ??? |
| `exploding_tensor` | ??? |
| `unchanged_tensor` | ??? |
| `loss_not_decreasing` | ??? |
| `dead_relu` | ??? |
| `confusion` | ??? |
| `overfit` | ??? |
| `tree_depth` | ??? |
| `tensor_variance` | ??? |
| `overtraining` | ??? |
| `poor_weight_initialization` | ??? |
| `saturated_activation` | ??? |
| `nlp_sequence_ratio` | ??? |
