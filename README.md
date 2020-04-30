# Amazon SageMaker Debugger
[![codecov](https://codecov.io/gh/awslabs/sagemaker-debugger/branch/master/graph/badge.svg)](https://codecov.io/gh/awslabs/sagemaker-debugger)
[![PyPI](https://badge.fury.io/py/smdebug.svg)](https://badge.fury.io/py/smdebug)

## Table of Contents

- [Overview](#overview)
- [SageMaker Debugger in action](#sagemaker-debugger-in-action)
- [Install](#install-sagemaker-debugger)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Further Documentation](#further-documentation)
- [Release Notes](#release-notes)


## Overview
[Amazon SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html) automates the debugging process of machine learning training jobs. From training jobs, Debugger allows you to
run your own training script (Zero Script Change experience) using Debugger built-in features&mdash;`Hook` and `Rule`&mdash;to capture tensors,
have flexibility to build customized Hooks and Rules for configuring tensors as you want,
and make the tensors available for analysis by saving in an [Amazon S3](https://aws.amazon.com/s3/?nc=sn&loc=0) bucket,
all through a flexible and powerful API.

The `smdebug` library powers Debugger by calling the saved tensors from the S3 bucket during the training job.
`smdebug` retrieves and filters the tensors generated from Debugger such as gradients, weights, and biases.

Debugger helps you develop better, faster, and cheaper models by minimally modifying estimator, tracing the tensors, catching anomalies while training models, and iterative model pruning.

Debugger supports TensorFlow, PyTorch, MXNet, and XGBoost frameworks.
The following list is a summary of the main functionalities of Debugger:

- Zero Script Change experience on SageMaker when using [supported containers](#support)
- Full visibility into any tensor part of the training process
- Real-time training job monitoring through Rules
- Automated anomaly detection and state assertions through built-in and custom Rules on SageMaker
- Actions on your training jobs based on the status of Rules
- Interactive exploration of saved tensors
- Distributed training support
- TensorBoard support

See [How it works](#how-it-works) for more details.


## SageMaker Debugger in Action
- Through the model pruning process using Debugger and `smdebug`, you can iteratively identify the importance of weights and cut neurons below a threshold you define. This process allows you to train the model with significantly fewer neurons, which means a lighter, more efficient, faster, and cheaper model without compromising accuracy.
![Debugger Iterative Model Pruning using ResNet](docs/resources/results_resnet.png?raw=true)
See [Using SageMaker Debugger and SageMaker Experiments for iterative model pruning](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/pytorch_iterative_model_pruning/iterative_model_pruning_resnet.ipynb) notebook for visualization and further information.
- Use Debugger with XGBoost in SageMaker Studio to save feature importance values and plot them in a notebook during training. ![Debugger XGBoost Visualization Example](docs/resources/xgboost_feature_importance.png?raw=true)
- Use Debugger with TensorFlow in SageMaker Studio to run built-in rules and visualize the loss. ![Debugger TensorFlow Visualization Example](docs/resources/tensorflow_rules_loss.png?raw=true)


## Install SageMaker Debugger

`smdebug` library runs on Python 3.x. Install `smdebug` through:

```
pip install smdebug
```

### Debugger Usage and Supported Frameworks
There are two ways in which you can enable SageMaker Debugger while training on SageMaker&mdash;Zero Script Change and Bring Your Own Training Container (BYOC).

#### Zero Script Change

You can use your own training script while using [AWS Deep Learning Containers (DLC)](https://aws.amazon.com/machine-learning/containers/) in TensorFlow, PyTorch, MXNet, and XGBoost frameworks. The AWS DLCs enable you to use Debugger with no changes to your training script by automatically adding SageMaker Debugger's `Hook`.
The following table shows currently supported versions of the four frameworks for Zero Script Change experience.

| Framework | Version |
| --- | --- |
| [TensorFlow](docs/tensorflow.md) | 1.15, 1.15.2, 2.1 |
| [MXNet](docs/mxnet.md) | 1.6 |
| [PyTorch](docs/pytorch.md) | 1.3, 1.4 |
| [XGBoost](docs/xgboost.md) | >=0.90-2 [As Built-in algorithm](xgboost.md#use-xgboost-as-a-built-in-algorithm)|

For the full list and information of the AWS DLCs, see [Deep Learning Containers Images](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html#deep-learning-containers-images-table).


#### Bring Your Own Training Container

`smdebug` supports frameworks other than the ones listed in the previous Zero Script Change section. You can use your own training script by adding a minimal modification.
Currently supported versions of frameworks are listed in the following table.

| Framework | Versions |
| --- | --- |
| [TensorFlow](docs/tensorflow.md) | 1.14. 1.15, 2.0.1, 2.1.0 |
| Keras (with TensorFlow backend) | 2.3 |
| [MXNet](docs/mxnet.md) | 1.4, 1.5, 1.6 |
| [PyTorch](docs/pytorch.md) | 1.2, 1.3, 1.4 |
| [XGBoost](docs/xgboost.md) | [As Framework](xgboost.md#use-xgboost-as-a-framework) |

### Support for Distributed Training and Known Limitations

<table>
    <thead>
        <tr>
           <th colspan=3>
           Distributed Training
           </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>Horovod</td>
            <td>Supported</td>
            <td>TF 1.15, PT 1.4, MX 1.6</td>
        </tr>
        <tr>
            <td>Not supported</td>
            <td>TF 2.x, PT 1.5</td>
        </tr>
        <tr>
            <td>Parameter Server-based</td>
            <td colspan=2>Not supported</td>
        </tr>
    </tbody>
</table>

## How It Works

Amazon SageMaker Debugger uses the construct of a `Hook` to save the values of requested tensors throughout the training process. You can then setup a `Rule` job which simultaneously monitors and validates these tensors to ensure
that training is progressing as expected.

A `Rule` checks for vanishing gradients, exploding tensor values, or poor weight initialization. Rules are attached to Amazon CloudWatch events, so that when a rule is triggered it changes the state of the CloudWatch event.
You can configure any action on the CloudWatch event, such as to stop the training job saving you time and money.

Debugger can be used inside or outside of SageMaker. However the built-in rules that AWS provides are only available for SageMaker training. Scenarios of usage can be classified into the following three cases.

#### Using SageMaker Debugger with Zero Script Change of Your Training Script

Here you specify which rules to use when setting up the estimator and run your existing script without no change. For an example of this, see [Running a Rule with Zero Script Change on SageMaker](#running-a-rule-with-zero-script-change-on-sageMaker).

#### Using SageMaker Debugger on Bring Your Own Container

You can use Debugger with your training script on your own container making only a minimal modification to your training script to add Debugger's `Hook`.
For an example template of code to use Debugger on your own container in TensorFlow 2.x frameworks, see [Running on Your Own Container](#Running-on-Your-Own-Container).
See the following instruction pages to set up Debugger in your preferred framework.
  - [TensorFlow](docs/tensorflow.md)
  - [MXNet](docs/mxnet.md)
  - [PyTorch](docs/pytorch.md)
  - [XGBoost](docs/xgboost.md)

#### Using SageMaker Debugger on a Non-SageMaker Environment

Here you write custom rules (or manually analyze the tensors) and modify your training script minimally to enable Debugger on a non-SageMaker Environment such as your local machine. For an example of this, see [Running Locally](#running-locally).

The reason for different setups is that Zero Script Change (via AWS Deep Learning Containers) uses custom framework forks of TensorFlow, PyTorch, MXNet, and XGBoost which add the `Hook` to the training job and save requested tensors automatically.
These framework forks are not available in custom containers or non-SageMaker environments, so you must modify your training script in these environments.


## Examples

### SageMaker Notebook Examples

To find a collection of demonstrations using Debugger, see [SageMaker Debugger Example Notebooks](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger).

#### Run a Rule with Zero Script Change

This example shows a how to use Debugger with Zero Script Change of
your training script on a SageMaker DLC.

```python
import sagemaker as sm
from sagemaker.debugger import rule_configs, Rule, CollectionConfig

# Choose a built-in rule to monitor your training job
rule = Rule.sagemaker(
    rule_configs.exploding_tensor(),
    # configure your rule if applicable
    rule_parameters={"tensor_regex": ".*"},
    # specify collections to save for processing your rule
    collections_to_save=[
        CollectionConfig(name="weights"),
        CollectionConfig(name="losses"),
    ],
)

# Pass the rule to the estimator
sagemaker_simple_estimator = sm.tensorflow.TensorFlow(
    entry_point="script.py", #replace script.py to your own training script
    role=sm.get_execution_role(),
    framework_version="1.15",
    py_version="py3",
    # argument for smdebug below
    rules=[rule],
)

sagemaker_simple_estimator.fit()
tensors_path = sagemaker_simple_estimator.latest_job_debugger_artifacts_path()

import smdebug.trials as smd
trial = smd.create_trial(out_dir=tensors_path)
print(f"Saved these tensors: {trial.tensor_names()}")
print(f"Loss values during evaluation were {trial.tensor('CrossEntropyLoss:0').values(mode=smd.modes.EVAL)}")
```

That's it! When you configure the `sagemaker_simple_estimator`,
you simply specify the `entry_point` to your training script python file.
When you run the `sagemaker_simple_estimator.fit()` API,
SageMaker will automatically monitor your training job for you with the Rules specified and create a `CloudWatch` event that tracks the status of the Rule,
so you can take any action based on them.

If you want additional configuration and control, see [Running SageMaker jobs with Debugger](docs/sagemaker.md) for more information.

#### Run Debugger in Your Own Container

The following example shows how to set `hook` to set a training model using Debugger in your own container.
This example is for containers in TensorFlow 2.x framework using GradientTape to configure the `hook`.

```python
import smdebug.tensorflow as smd
hook = smd.KerasHook(out_dir=args.out_dir)

model = tf.keras.models.Sequential([ ... ])
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            # wrap the tape to capture tensors
            with hook.wrap_tape(tf.GradientTape(persistent=True)) as tape:
                logits = model(data, training=True)  # (32,10)
                loss_value = cce(labels, logits)
            grads = tape.gradient(loss_value, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            acc = train_acc_metric(dataset_labels, logits)
            # manually save metric values
            hook.record_tensor_value(tensor_name="accuracy", tensor_value=acc)
```

To see a full script of this, refer to the [tf_keras_gradienttape.py](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf_keras_gradienttape.py) example script.
For a notebook example of using BYOC in PyTorch, see [Using Amazon SageMaker Debugger with Your Own PyTorch Container](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-debugger/pytorch_custom_container/pytorch_byoc_smdebug.ipynb)

#### Run Debugger Locally
Requires Python 3.6+ and this example uses tf.keras.

To use Debugger, simply add a callback `hook`:
```python
import smdebug.tensorflow as smd
hook = smd.KerasHook(out_dir='~/smd_outputs/')

model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)

# Add the hook as a callback
model.fit(x_train, y_train, epochs=2, callbacks=[hook])
model.evaluate(x_test, y_test, callbacks=[hook])

# Create a trial to inspect the saved tensors
trial = smd.create_trial(out_dir='~/smd_outputs/')
print(f"Saved these tensors: {trial.tensor_names()}")
print(f"Loss values during evaluation were {trial.tensor('CrossEntropyLoss:0').values(mode=smd.modes.EVAL)}")
```

## Further Documentation

| Section | Description |
| --- | --- |
| [SageMaker Training](docs/sagemaker.md) | SageMaker users, we recommend you start with this page on how to run SageMaker training jobs with SageMaker Debugger |
| Frameworks <ul><li>[TensorFlow](docs/tensorflow.md)</li><li>[PyTorch](docs/pytorch.md)</li><li>[MXNet](docs/mxnet.md)</li><li>[XGBoost](docs/xgboost.md)</li></ul> | See the frameworks pages for details on what's supported and how to modify your training script if applicable |
<<<<<<< HEAD
| [APIs for Saving Tensors](docs/api.md) | Full description of the APIs for saving tensors |
| [Programming Model for Analysis](docs/analysis.md) | For description of the programming model provided by the APIs that enable you to perform interactive exploration of tensors saved as well as to write your own Rules monitoring your training jobs. |
=======
| [APIs for Saving Tensors](docs/api.md) | Full description of our APIs on saving tensors |
| [Programming Model for Analysis](docs/analysis.md) | For description of the programming model provided by the APIs that enable you to perform interactive exploration of tensors saved, as well as to write your own Rules monitoring your training jobs. |
>>>>>>> 15ed2dc... Update README.md

## Release Notes

### [Latest release v0.7.2](https://github.com/awslabs/sagemaker-debugger/releases)

   - Introducing experimental support for TF 2.x training scripts using GradientTape -
   With this update, weights, bias, loss, metrics, and gradients are captured by SageMaker Debugger.
   GradientTape in TF 2.x captures these tensors from custom training jobs. An example of GradientTape implementation to a custom ResNet training script using TensorFlow's Keras interface is provided at [tf_keras_gradienttape.py](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf_keras_gradienttape.py).
   GradientTape does not work with Zero Script Change experience at this time.

        *Note*: Training scripts using GradientTape for higher-order gradients or multiple tapes are not supported. Distributed training scripts that use GradientTape are not supported at this time.

   - Support `SyncOnReadVariable` in mirrored strategy - Fixes a bug that occurred because the `SyncOnRead` distributed variable was not supported with `smdebug`. Also enables the use of `smdebug` with training scripts using TF 2.x MirroredStrategy with the `fit()` API.

   - Turn off hook and write only from one worker for unsupported distributed training techniques – Fixes a crash when distributed training in PyTorch framework is implemented using generic multiprocessing library, which is not a method supported by `smdebug`. This fix handles this case and ensures that tensors are saved.

   - Bug fix: Pytorch: Register only if tensors require gradients – Users were observing a crash when training with pre-trained embeddings which does not need gradient updates. This fix checks if a gradient update is required and registers a backward hook only in those cases.

## License
This library is licensed under the Apache 2.0 License.
