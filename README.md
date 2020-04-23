# Amazon SageMaker Debugger
[![codecov](https://codecov.io/gh/awslabs/sagemaker-debugger/branch/master/graph/badge.svg)](https://codecov.io/gh/awslabs/sagemaker-debugger)
[![PyPI](https://badge.fury.io/py/smdebug.svg)](https://badge.fury.io/py/smdebug)

## Table of Contents

- [Overview](#overview)
- [Support](#support)
- [How It Works](#how-it-works)
- [Docs](#docs)
- [Examples](#examples)
- [SageMaker Debugger in action](#sagemaker-debugger-in-action)

## Overview
[Amazon SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html) automates the debugging process of machine learning training jobs. It allows you to save tensors from training jobs and makes these tensors available for analysis, all through a flexible and powerful API.
`smdebug` library powers [Amazon SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html) by calling the saved tensors in S3 during the training job. `smdebug` retrieves and filters tensors such as gradients, weights, and biases.

Amazon SageMaker Debugger supports TensorFlow, PyTorch, MXNet, and XGBoost frameworks.
The following list is a summary of the main functionalities of Amazon SageMaker Debugger:

- Zero Script Change experience on SageMaker when using [supported containers](#support)
- Full visibility into any tensor part of the training process
- Real-time training job monitoring through Rules
- Automated anomaly detection and state assertions through built-in and custom Rules on SageMaker
- Actions on your training jobs based on the status of Rules
- Interactive exploration of saved tensors
- Distributed training support
- TensorBoard support

## Support

### [Latest release v0.7.2](https://github.com/awslabs/sagemaker-debugger/releases)

   - Experimental support for TF 2.x GradientTape - Introducing experimental support for TF 2.x training scripts using GradientTape.
   SageMaker Debugger GradientTape now captures tensors such as loss, metrics, weights, biases, and gradients by modifying training scripts in TF 2.x framework. A sample training script is provided [here](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf_keras_gradienttape.py).
   GradientTape does not work with Zero Script Change experience at this time.

        Note: Training scripts using GradientTape for higher-order gradients or multiple tapes are not supported. Distributed training scripts that use GradientTape are not supported at this time.

   - Support SyncOnReadVariable in mirrored strategy - Fixes a bug that occurred because SyncOnRead distributed variable was not supported with smdebug. Also enables the use of smdebug with training scripts using TF 2.x MirroredStrategy with fit() API.

   - Turn off hook and write only from one worker for unsupported distributed training techniques – Fixes a crash when distributed training in PyTorch framework is implemented using generic multiprocessing library, which is not a method supported by smdebug. This fix handles this case and ensures that tensors are saved.

   - Bug fix: Pytorch: Register only if tensors require gradients – Users were observing a crash when training with pre-trained embeddings which does not need gradient updates. This fix checks if a gradient update is required and registers a backward hook only in those cases.

#### Setting up SageMaker Debugger

- `smdebug` library runs on Python 3.x. Install `smdebug` through:

    ```
    pip install smdebug
    ```

#### Zero Script Change

| Framework | Version |
| --- | --- |
| [TensorFlow](tensorflow.md) | 1.15, 2.1 |
| [MXNet](mxnet.md) | 1.6 |
| [PyTorch](pytorch.md) | 1.3, 1.4 |
| [XGBoost](xgboost.md) | >=0.90-2 [As Built-in algorithm](xgboost.md#use-xgboost-as-a-built-in-algorithm)|

#### Bring your own training container

| Framework | Versions |
| --- | --- |
| [TensorFlow](tensorflow.md) | 1.14. 1.15, 2.0.1, 2.1.0 |
| Keras (with TensorFlow backend) | 2.3 |
| [MXNet](mxnet.md) | 1.4, 1.5, 1.6 |
| [PyTorch](pytorch.md) | 1.2, 1.3, 1.4 |
| [XGBoost](xgboost.md) | [As Framework](xgboost.md#use-xgboost-as-a-framework) |

#### Support for Distributed Training and Known Limitations

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
A rule might check for vanishing gradients, or exploding tensor values, or poor weight initialization. Rules are attached to CloudWatch events, so that when a rule is triggered it changes the state of the CloudWatch event. You can configure any action on the CloudWatch event, such as to stop the training job saving you time and money.

Amazon SageMaker Debugger can be used inside or outside of SageMaker. However the built-in rules that AWS provides are only available for SageMaker training. Scenarios of usage can be classified into the following:
- **SageMaker Zero-Script-Change**: Here you specify which rules to use when setting up the estimator and run your existing script, no changes needed. See [the first example below](#running-a-rule-with-zero-script-change-on-sageMaker).
- **SageMaker Bring-Your-Own-Container**: Here you specify the rules to use, and modify your training script minimally to enable SageMaker Debugger.
- **Non-SageMaker**: Here you write custom rules (or manually analyze the tensors) and modify your training script minimally to enable SageMaker Debugger. See [the second example below](#running-locally).

The reason for different setups is that SageMaker Zero-Script-Change (via AWS Deep Learning Containers) uses custom framework forks of TensorFlow, PyTorch, MXNet, and XGBoost which add our Hook to the training job and save requested tensors automatically.
These framework forks are not available in custom containers or non-SM environments, so you must modify your training script in these environments.

## Docs

| Section | Description |
| --- | --- |
| [SageMaker Training](docs/sagemaker.md) | SageMaker users, we recommend you start with this page on how to run SageMaker training jobs with SageMaker Debugger |
| Frameworks <ul><li>[TensorFlow](docs/tensorflow.md)</li><li>[PyTorch](docs/pytorch.md)</li><li>[MXNet](docs/mxnet.md)</li><li>[XGBoost](docs/xgboost.md)</li></ul> | See the frameworks pages for details on what's supported and how to modify your training script if applicable |
| [APIs for Saving Tensors](docs/api.md) | Full description of our APIs on saving tensors |
| [Programming Model for Analysis](docs/analysis.md) | For description of the programming model provided by our APIs which allows you to perform interactive exploration of tensors saved as well as to write your own Rules monitoring your training jobs. |

## Examples
### Notebooks
Example notebooks demonstrating different functionalities of SageMaker Debugger are provided [here](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger).

### Running a Rule with Zero Script Change on SageMaker
This example uses a zero-script-change experience, where you can use your training script as-is. Refer to [Running SageMaker jobs with Amazon SageMaker Debugger](docs/sagemaker.md) for more details on this.
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
    entry_point="script.py",
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

That's it! Amazon SageMaker will automatically monitor your training job for you with the Rules specified and create a CloudWatch
event which tracks the status of the Rule, so you can take any action based on them.

If you want greater configuration and control, we offer that too. Head over [here](docs/sagemaker.md) for more information.

### Running Locally
Requires Python 3.6+, and this example uses tf.keras.

To use Amazon SageMaker Debugger, simply add a callback hook:
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

## SageMaker Debugger in action
- Using SageMaker Debugger with XGBoost in SageMaker Studio to save feature importance values and plot them in a notebook during training. ![](docs/resources/xgboost_feature_importance.png?raw=true)
- Using SageMaker Debugger with TensorFlow in SageMaker Studio to run built-in rules and visualize the loss. ![](docs/resources/tensorflow_rules_loss.png?raw=true)



## License
This library is licensed under the Apache 2.0 License.
