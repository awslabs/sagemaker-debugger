# SageMaker

There are two cases for SageMaker:
- Zero-Code-Change (ZCC): Here you specify which rules to use, and run your existing script.
    - Supported in Deep Learning Containers: `TensorFlow==1.15, PyTorch==1.3, MXNet==1.6`
- Bring-Your-Own-Container (BYOC): Here you specify the rules to use, and modify your training script.
    - Supported with `TensorFlow==1.13/1.14/1.15, PyTorch==1.2/1.3, MXNet==1.4/1.5/1.6`

The reason for different setups is that SageMaker Zero-Script-Change (via Deep Learning Containers) uses custom framework forks of TensorFlow, PyTorch, MXNet, and XGBoost to save tensors automatically.
These framework forks are not available in custom containers or non-SM environments, so you must modify your training script in these environments.

## Configuration Details

This configuration is used for both ZCC and BYOC. The only difference is that with a custom container, you modify your training script as well. See the framework pages below for details on how to modify your training script.

- [TensorFlow](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/tensorflow.md)
- [PyTorch](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/pytorch.md)
- [MXNet](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/mxnet.md)
- [XGBoost](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/xgboost.md)

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

A full example script is below:
```python
import sagemaker
from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig, TensorBoardOutputConfig, CollectionConfig

hook_parameters = {
    "include_regex": "my_regex,another_regex", # comma-separated string of regexes
    "save_interval": 100,
    "save_steps": "1,2,3,4", # comma-separated string of steps to save
    "start_step": 1,
    "end_step": 2000,
    "reductions": "min,max,mean,std,abs_variance,abs_sum,abs_l2_norm",
}
weights_config = CollectionConfiguration("weights")
biases_config = CollectionConfiguration("biases")
losses_config = CollectionConfiguration("losses")
tb_config = TensorBoardOutputConfig(s3_output_path="s3://my-bucket/tensorboard")

hook_config = DebuggerHookConfig(
    s3_output_path="s3://my-bucket/smdebug",
    hook_parameters=hook_parameters,
    collection_configs=[weights_config, biases_config, losses_config],
)

exploding_tensor_rule = Rule.sagemaker(
    base_config=rule_configs.exploding_tensor(),
    rule_parameters={
        "tensor_regex": ".*",
    },
    collections_to_save=[weights_config, losses_config],
)
vanishing_gradient_rule = Rule.sagemaker(base_config=rule_configs.vanishing_gradient())

# Or use sagemaker.pytorch.PyTorch or sagemaker.mxnet.MXNet
sagemaker_simple_estimator = sagemaker.tensorflow.TensorFlow(
    entry_point=simple_entry_point_script,
    role=sagemaker.get_execution_role(),
    base_job_name=args.job_name,
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="1.15",
    py_version="py3",
    # smdebug-specific arguments below
    rules=[exploding_tensor_rule, vanishing_gradient_rule],
    debugger_hook_config=hook_config,
    tensorboard_output_config=tb_config,
)

sagemaker_simple_estimator.fit()
```




## List of Builtin Rules
Full list of rules is:
| Rule Name | Behavior |
|---|---|
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
