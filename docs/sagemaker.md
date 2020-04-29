## Running SageMaker jobs with Amazon SageMaker Debugger

## Outline
- [Configuring SageMaker Debugger](#configuring-sagemaker-debugger)
  - [Saving data](#saving-data)
    - [Saving built-in collections that we manage](#saving-built-in-collections-that-we-manage)
    - [Saving reductions for a custom collection](#saving-reductions-for-a-custom-collection)
    - [Enabling TensorBoard summaries](#enabling-tensorboard-summaries)
  - [Rules](#rules)
    - [Built In Rules](#built-in-rules)
    - [Custom Rules](#custom-rules)
- [Interactive Exploration](#interactive-exploration)
- [SageMaker Studio](#sagemaker-studio)
- [TensorBoard Visualization](#tensorboard-visualization)
- [Example Notebooks](#example-notebooks)

## Configuring SageMaker Debugger

Regardless of which of the two above ways you have enabled SageMaker Debugger, you can configure it using the SageMaker python SDK. There are two aspects to this configuration.
- You can specify what tensors to be saved, when they should be saved and in what form they should be saved.
- You can specify which Rule you want to monitor your training job with. This can be either a built in rule that SageMaker provides, or a custom rule that you can write yourself.

### Saving Data

SageMaker Debugger gives you a powerful and flexible API to save the tensors you choose at the frequencies you want. These configurations are made available in the SageMaker Python SDK through the `DebuggerHookConfig` class.

#### Saving built-in collections that we manage
Learn more about these built in collections [here](api.md).

```python
from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
hook_config = DebuggerHookConfig(
    s3_output_path='s3://smdebug-dev-demo-pdx/mnist',
    hook_parameters={
        "save_interval": 100
    },
    collection_configs=[
        CollectionConfig("weights"),
        CollectionConfig("gradients"),
        CollectionConfig("losses"),
        CollectionConfig(
            name="biases",
            parameters={
                "save_interval": 10,
                "end_step": 500
            }
        ),
    ]
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
    debugger_hook_config=hook_config
)
sagemaker_estimator.fit()
```

#### Saving reductions for a custom collection
You can define your collection of tensors. You can also choose to save certain reductions of tensors only instead of saving the full tensor. You may choose to do this to reduce the amount of data saved. Please note that when you save reductions, unless you pass the flag `save_raw_tensor`, only these reductions will be available for analysis. The raw tensor will not be saved.

```python
from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
hook_config = DebuggerHookConfig(
    s3_output_path='s3://smdebug-dev-demo-pdx/mnist',
    collection_configs=[
        CollectionConfig(
            name="activations",
            parameters={
                "include_regex": "relu|tanh",
                "reductions": "mean,variance,max,abs_mean,abs_variance,abs_max"
            })
    ]
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
    debugger_hook_config=hook_config
)
sagemaker_estimator.fit()
```

#### Enabling TensorBoard summaries
SageMaker Debugger can automatically generate tensorboard scalar summaries,
distributions and histograms for tensors saved. This can be enabled by
passing a `TensorBoardOutputConfig` object when creating an Estimator as follows.
You can also choose to disable or enable histograms specifically for different collections.
By default a collection has `save_histogram` flag set to True.
Note that scalar summaries are added to TensorBoard for all `ScalarCollections` and any scalar saved through `hook.save_scalar`.
Refer [API](api.md) for more details on scalar collections and `save_scalar` method.

The below example saves weights and gradients as full tensors, and also saves the gradients as histograms and distributions to visualize in TensorBoard.
These will be saved to the location passed in `TensorBoardOutputConfig` object.
```python
from sagemaker.debugger import DebuggerHookConfig, CollectionConfig, TensorBoardOutputConfig
hook_config = DebuggerHookConfig(
    s3_output_path='s3://smdebug-dev-demo-pdx/mnist',
    collection_configs=[
        CollectionConfig(
            name="weights",
            parameters={"save_histogram": False}),
        CollectionConfig(name="gradients"),
    ]
)

tb_config = TensorBoardOutputConfig('s3://smdebug-dev-demo-pdx/mnist/tensorboard')

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
    debugger_hook_config=hook_config,
    tensorboard_output_config=tb_config
)
sagemaker_estimator.fit()
```

For more details, refer our [API page](api.md).

### Rules
Here are some examples on how to run Rules with your training jobs.

Note that passing a `CollectionConfig` object to the Rule as `collections_to_save`
is equivalent to passing it to the `DebuggerHookConfig` object as `collection_configs`.
This is just a shortcut for your convenience.

#### Built in Rules
The Built-in Rules, or SageMaker Rules, are described in detail on [this page](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html)


Scope of Validity | Rules |
|---|---|
| Generic Deep Learning models (TensorFlow, Apache MXNet, and PyTorch) |<ul><li>[`dead_relu`](https://docs.aws.amazon.com/sagemaker/latest/dg/dead-relu.html)</li><li>[`exploding_tensor`](https://docs.aws.amazon.com/sagemaker/latest/dg/exploding-tensor.html)</li><li>[`poor_weight_initialization`](https://docs.aws.amazon.com/sagemaker/latest/dg/poor-weight-initialization.html)</li><li>[`saturated_activation`](https://docs.aws.amazon.com/sagemaker/latest/dg/saturated-activation.html)</li><li>[`vanishing_gradient`](https://docs.aws.amazon.com/sagemaker/latest/dg/vanishing-gradient.html)</li><li>[`weight_update_ratio`](https://docs.aws.amazon.com/sagemaker/latest/dg/weight-update-ratio.html)</li></ul> |
| Generic Deep learning models (TensorFlow, MXNet, and PyTorch) and the XGBoost algorithm | <ul><li>[`all_zero`](https://docs.aws.amazon.com/sagemaker/latest/dg/all-zero.html)</li><li>[`class_imbalance`](https://docs.aws.amazon.com/sagemaker/latest/dg/class-imbalance.html)</li><li>[`confusion`](https://docs.aws.amazon.com/sagemaker/latest/dg/confusion.html)</li><li>[`loss_not_decreasing`](https://docs.aws.amazon.com/sagemaker/latest/dg/loss-not-decreasing.html)</li><li>[`overfit`](https://docs.aws.amazon.com/sagemaker/latest/dg/overfit.html)</li><li>[`overtraining`](https://docs.aws.amazon.com/sagemaker/latest/dg/overtraining.html)</li><li>[`similar_across_runs`](https://docs.aws.amazon.com/sagemaker/latest/dg/similar-across-runs.html)</li><li>[`tensor_variance`](https://docs.aws.amazon.com/sagemaker/latest/dg/tensor-variance.html)</li><li>[`unchanged_tensor`](https://docs.aws.amazon.com/sagemaker/latest/dg/unchanged-tensor.html)</li></ul>|
| Deep learning applications |<ul><li>[`check_input_images`](https://docs.aws.amazon.com/sagemaker/latest/dg/checkinput-mages.html)</li><li>[`nlp_sequence_ratio`](https://docs.aws.amazon.com/sagemaker/latest/dg/nlp-sequence-ratio.html)</li></ul> |
| XGBoost algorithm | <ul><li>[`tree_depth`](https://docs.aws.amazon.com/sagemaker/latest/dg/tree-depth.html)</li></ul>|


#### Running built-in SageMaker Rules
You can run a SageMaker built-in Rule as follows using the `Rule.sagemaker` method.
The first argument to this method is the base configuration that is associated with the Rule.
We configure them as much as possible.
You can take a look at the ruleconfigs that we populate for all built-in rules [here](https://github.com/awslabs/sagemaker-debugger-rulesconfig).
You can choose to customize these parameters using the other parameters.

These rules are run on our pre-built Docker images which are listed [here](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-docker-images-rules.html).
You are not charged for the instances when running SageMaker built-in rules.

A list of all our built-in rules are provided [below](#built-in-rules).
```python
from sagemaker.debugger import Rule, CollectionConfig, rule_configs

exploding_tensor_rule = Rule.sagemaker(
    base_config=rule_configs.exploding_tensor(),
    rule_parameters={"collection_names": "weights,losses"},
    collections_to_save=[
        CollectionConfig("weights"),
        CollectionConfig("losses")
    ]
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
#### Custom Rules

You can write your own rule custom made for your application and provide it, so SageMaker can monitor your training job using your rule. To do so, you need to understand the programming model that `smdebug` provides. Our page on [Programming Model for Analysis](analysis.md) describes the APIs that we provide to help you write your own rule.
Please refer to [this example notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/tensorflow_keras_custom_rule/tf-keras-custom-rule.ipynb) for a demonstration of creating your custom rule and running it on SageMaker.

#### Running custom Rules
To run a custom rule, you have to provide a few additional parameters.
Key parameters to note are a file which has the implementation of your Rule class `source`,
 the name of the Rule class (`rule_to_invoke`), the type of instance to run the Rule job on (`instance_type`),
 the size of the volume on that instance (`volume_size_in_gb`), and the docker image to use for running this job (`image_uri`).

Please refer to the documentation [here](https://github.com/aws/sagemaker-python-sdk/blob/391733efd433c5e26afb56102c76ab7472f94b3d/src/sagemaker/debugger.py#L190) for more details.

We have pre-built Docker images that you can use to run your custom rules.
These are listed [here](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-docker-images-rules.html).
You can also choose to build your own Docker image for custom rule evaluation.
Please refer to the repository [SageMaker Debugger Rules Container](https://github.com/awslabs/sagemaker-debugger-rules-container) for instructions on how to build such an image.

```python
from sagemaker.debugger import Rule, CollectionConfig

custom_coll = CollectionConfig(
    name="relu_activations",
    parameters={
        "include_regex": "relu",
        "save_interval": 500,
        "end_step": 5000
    })
improper_activation_rule = Rule.custom(
    name='improper_activation_job',
    image_uri='552407032007.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-debugger-rule-evaluator:latest',
    instance_type='ml.c4.xlarge',
    volume_size_in_gb=400,
    source='rules/custom_rules.py',
    rule_to_invoke='ImproperActivation',
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
    rules=[improper_activation_rule],
)
sagemaker_estimator.fit()
```

For more details, refer our [Analysis page](analysis.md).

## Interactive Exploration

`smdebug` SDK also allows you perform interactive and real-time exploration of the data saved. You can choose to inspect the tensors saved, or visualize them through your custom plots.
You can retrieve these tensors as numpy arrays allowing you to use your favorite analysis libraries right in a SageMaker notebook instance. We have couple of example notebooks demonstrating this.
- [Real-time anaysis in a notebook during training](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/mxnet_realtime_analysis/mxnet-realtime-analysis.ipynb)
- [Interactive tensor analysis in a notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/mnist_tensor_analysis/mnist_tensor_analysis.ipynb)

## SageMaker Studio

SageMaker Debugger is on by default for supported training jobs on the official SageMaker Framework containers (or AWS Deep Learning Containers) during SageMaker training jobs.
In this default scenario, SageMaker Debugger takes the losses and metrics from your training job and publishes them to SageMaker Metrics, allowing you to track these metrics in SageMaker Studio.
You can also see the status of Rules you have enabled for your training job right in the Studio. [Here](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-visualization.html) are screenshots of that experience.

## TensorBoard Visualization

If you have enabled TensorBoard outputs for your training job through SageMaker Debugger, TensorBoard artifacts will automatically be generated for the tensors saved.
You can then point your TensorBoard instance to that S3 location and review the visualizations for the tensors saved.

## Example Notebooks

We have a bunch of [example notebooks](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger) here demonstrating different aspects of SageMaker Debugger.
