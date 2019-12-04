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

##### Running built-in SageMaker Rules
```python
from sagemaker.debugger import Rule, rule_configs

weights_collection = rule_configs.get_collection('weights')
losses_collection = rule_configs.get_collection('losses')

exploding_tensor_rule = Rule.sagemaker(
    base_config=rule_configs.exploding_tensor(),
    rule_parameters={"collection_names": "weights,losses"},
    collections_to_save=[weights_collection, losses_collection]
)

vanishing_gradient_rule = Rule.sagemaker(
    base_config=rule_configs.vanishing_gradient()
)

import sagemaker as sm
sagemaker_estimator = sm.tensorflow.TensorFlow(
    entry_point='src/mnist.py',
    role=sm.get_execution_role(),
    base_job_name='smdebug-demo-job',
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="1.15",
    py_version="py3",
    # smdebug-specific arguments below
    rules=[exploding_tensor_rule, vanishing_gradient_rule],
)
sagemaker_estimator.fit()
```

##### Running custom Rules
```python
from sagemaker.debugger import Rule, CollectionConfig

custom_coll = CollectionConfig(
    name="relu_activations",
    parameters={
        "include_regex": "relu",
        "save_interval": 500,
        "end_step": 5000
    })
saturated_activation_rule = Rule.sagemaker(
    name='saturated_activation',
    rule_parameters={"collection_names": "relu_activations"},
    collections_to_save=[custom_coll]
)

import sagemaker as sm
sagemaker_estimator = sm.tensorflow.TensorFlow(
    entry_point='src/mnist.py',
    role=sm.get_execution_role(),
    base_job_name='smdebug-demo-job',
    train_instance_count=1,
    train_instance_type="ml.m4.xlarge",
    framework_version="1.15",
    py_version="py3",
    # smdebug-specific arguments below
    rules=[saturated_activation_rule],
)
sagemaker_estimator.fit()
```

Hopefully the above examples gave you an overview of how you can enable Rules when training on SageMaker. We'll cover this in more detail in the below sections.

### Saving Data

SageMaker Debugger gives you a powerful and flexible API to save the tensors you choose at the frequencies you want. These configurations are made available in the SageMaker Python SDK through the `DebuggerHookConfig` class. Which has the following signature:

```
DebuggerHookConfig(
    s3_output_path=None,
    container_local_output_path=None,
    hook_parameters=None,
    collection_configs=None,
    ):
```

### Rules

#### Built in Rules
The Built-in Rules, or SageMaker Rules, are described in detail on [this page](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html)


Scope of Validity | Rules |
|---|---|
| Generic Deep Learning models (TensorFlow, Apache MXNet, and PyTorch) |<ul><li>[`dead_relu`](https://docs.aws.amazon.com/sagemaker/latest/dg/dead-relu.html)</li><li>[`exploding_tensor`](https://docs.aws.amazon.com/sagemaker/latest/dg/exploding-tensor.html)</li><li>[`poor_weight_initialization`](https://docs.aws.amazon.com/sagemaker/latest/dg/poor-weight-initialization.html)</li><li>[`saturated_activation`](https://docs.aws.amazon.com/sagemaker/latest/dg/saturated-activation.html)</li><li>[`vanishing_gradient`](https://docs.aws.amazon.com/sagemaker/latest/dg/vanishing-gradient.html)</li><li>[`weight_update_ratio`](https://docs.aws.amazon.com/sagemaker/latest/dg/weight-update-ratio.html)</li></ul> |
| Generic Deep learning models (TensorFlow, MXNet, and PyTorch) and the XGBoost algorithm | <ul><li>[`all_zero`](https://docs.aws.amazon.com/sagemaker/latest/dg/all-zero.html)</li><li>[`class_imbalance`](https://docs.aws.amazon.com/sagemaker/latest/dg/class-imbalance.html)</li><li>[`confusion`](https://docs.aws.amazon.com/sagemaker/latest/dg/confusion.html)</li><li>[`loss_not_decreasing`](https://docs.aws.amazon.com/sagemaker/latest/dg/loss-not-decreasing.html)</li><li>[`overfit`](https://docs.aws.amazon.com/sagemaker/latest/dg/overfit.html)</li><li>[`overtraining`](https://docs.aws.amazon.com/sagemaker/latest/dg/overtraining.html)</li><li>[`similar_across_runs`](https://docs.aws.amazon.com/sagemaker/latest/dg/similar-across-runs.html)</li><li>[`tensor_variance`](https://docs.aws.amazon.com/sagemaker/latest/dg/tensor-variance.html)</li><li>[`unchanged_tensor`](https://docs.aws.amazon.com/sagemaker/latest/dg/unchanged-tensor.html)</li>/ul>|
| Deep learning applications |<ul><li>[`check_input_images`](https://docs.aws.amazon.com/sagemaker/latest/dg/checkinput-mages.html)</li><li>[`nlp_sequence_ratio`](https://docs.aws.amazon.com/sagemaker/latest/dg/nlp-sequence-ratio.html)</li></ul> |
| XGBoost algorithm | <ul><li>[`tree_depth`](https://docs.aws.amazon.com/sagemaker/latest/dg/tree-depth.html)</li></ul>|


#### Custom Rules

You can write your own rule custom made for your application and provide it, so SageMaker can monitor your training job using your rule. To do so, you need to understand the programming model that `smdebug` provides. Our page on [Programming Model for Analysis](analysis.md) describes the APIs that we provide to help you write your own rule.

### Interactive Exploration

### SageMaker Studio

### TensorBoard Visualization
