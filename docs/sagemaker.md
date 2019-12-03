## Running SageMaker jobs with Amazon SageMaker Debugger

## Outline
- [Enabling SageMaker Debugger](#enabling-sagemaker-debugger)
  - [Zero Script Change](#zero-script-change)
  - [Bring your own training container](#bring-your-own-training-container)
- [Configuring SageMaker Debugger](#configuring-sagemaker-debugger)
  - [Examples](#examples)
  - [Saving data](#saving-data)
  - [Rules](#rules)
    - [Built In Rules](#built-in-rules)
    - [Custom Rules](#custom-rules)

## Enabling SageMaker Debugger
There are two ways in which you can enable SageMaker Debugger while training on SageMaker.

### Zero Script Change
We have equipped the official Framework containers on SageMaker with custom versions of supported frameworks TensorFlow, PyTorch, MXNet and XGBoost. These containers enable you to use SageMaker Debugger with no changes to your training script, by automatically adding [SageMaker Debugger's Hook](api.md#glossary).

Here's a list of frameworks and versions which support this experience.

| Framework | Version |
| --- | --- |
| [TensorFlow](tensorflow.md) | 1.15 |
| [MXNet](mxnet.md) | 1.6 |
| [PyTorch](pytorch.md) | 1.3 |
| [XGBoost](xgboost.md) | |

More details for the deep learning frameworks on which containers these are can be found here: [SageMaker Framework Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) and [AWS Deep Learning Containers](https://aws.amazon.com/machine-learning/containers/). You do not have to specify any training container image if you want to use them on SageMaker. You only need to specify the version above to use these containers.

### Bring your own training container

This library `smdebug` itself supports versions other than the ones listed above. If you want to use SageMaker Debugger with a version different from the above, you will have to orchestrate your training script with a few lines. Before we discuss how these changes look like, let us take a look at the versions supported.

| Framework | Versions |
| --- | --- |
| [TensorFlow](tensorflow.md) | 1.13, 1.14, 1.15 |
| Keras (with TensorFlow backend) | 2.3 |
| [MXNet](mxnet.md) | 1.4, 1.5, 1.6 |
| [PyTorch](pytorch.md) | 1.2, 1.3 |
| [XGBoost](xgboost.md) | |

#### Setting up SageMaker Debugger with your script on your container

- Ensure that you are using Python3 runtime as `smdebug` only supports Python3.
- Install `smdebug` binary through `pip install smdebug`
- Make some minimal modifications to your training script to add SageMaker Debugger's Hook. Please refer to the framework pages linked below for instructions on how to do that.
    - [TensorFlow](tensorflow.md)
    - [PyTorch](pytorch.md)
    - [MXNet](mxnet.md)
    - [XGBoost](xgboost.md)

## Configuring SageMaker Debugger

Regardless of which of the two above ways you have enabled SageMaker Debugger, you can configure it using the SageMaker python SDK. There are two aspects to this configuration.
- You can specify which Rule you want to monitor your training job with. This can be either a built in rule that SageMaker provides, or a custom rule that you can write yourself.
- You can specify what tensors to be saved, when they should be saved and in what form they should be saved.

Let us start by reviewing examples for a few scenarios before going into them in detail.

### Examples

##### Running a built-in rule
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




## Built-in Rules
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
