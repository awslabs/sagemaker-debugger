# Amazon SageMaker Debugger

- [Overview](#overview)
- [Examples](#sagemaker-example)
- [How It Works](#how-it-works)
- [Docs](#docs)

## Overview
Amazon SageMaker Debugger is an offering from AWS which help you automate the debugging of machine learning training jobs.
This library powers Amazon SageMaker Debugger, and helps you develop better, faster and cheaper models by catching common errors quickly.
It allows you to save tensors from training jobs and makes these tensors available for analysis, all through a flexible and powerful API.
It supports TensorFlow, PyTorch, MXNet, and XGBoost on Python 3.6+.

- Zero Script Change experience on SageMaker when using supported versions of SageMaker Framework containers or AWS Deep Learning containers
- Full visibility into any tensor part of the training process
- Real-time training job monitoring through Rules
- Automated anomaly detection and state assertions
- Interactive exploration of saved tensors
- Actions on your training jobs based on the status of Rules
- Distributed training support
- TensorBoard support

## Examples
### Notebooks
We have a bunch of [example notebooks](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger) here demonstrating different functionality of SageMaker Debugger.

### Running a Rule with Zero Script Change on SageMaker
This example uses a zero-script-change experience, where you can use your training script as-is. Refer [Running SageMaker jobs with Amazon SageMaker Debugger](docs/sagemaker.md) for more details on this.
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

import smdebug as smd
trial = smd.trials.create_trial(out_dir=tensors_path)
print(f"Saved these tensors: {trial.tensor_names()}")
print(f"Loss values during evaluation were {trial.tensor('CrossEntropyLoss:0').values(mode=smd.modes.EVAL)}")
```

That's it! Amazon SageMaker will automatically monitor your training job for you with the Rules specified and create a CloudWatch
event which tracks the status of the Rule, so you can take any action based on them.

If you want greater configuration and control, we offer that too. Head over [here](docs/sagemaker.md) for more information.

### Running Locally
Requires Python 3.6+, and this example uses tf.keras. Run
```
pip install smdebug
```

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

## How It Works

Amazon SageMaker Debugger uses a `Hook` to store the values of tensors throughout the training process.
Another process called a `Rule` job simultaneously monitors and validates these outputs to ensure
that training is progressing as expected.
A rule might check for vanishing gradients, or exploding tensor values, or poor weight initialization.
If a rule is triggered, it will raise a CloudWatch event, saving you time and money.

Amazon SageMaker Debugger can be used inside or outside of SageMaker. There are three main use cases:
- SageMaker Zero-Script-Change: Here you specify which rules to use when setting up the estimator and run your existing script, no changes needed. See the first example above.
- SageMaker Bring-Your-Own-Container: Here you specify the rules to use, and modify your training script.
- Non-SageMaker: Here you write custom rules (or manually analyze the tensors) and modify your training script. See the second example above.

The reason for different setups is that SageMaker Zero-Script-Change (via Deep Learning Containers) uses custom framework forks of TensorFlow, PyTorch, MXNet, and XGBoost to save tensors automatically.
These framework forks are not available in custom containers or non-SM environments, so you must modify your training script in these environments.

## Docs

Section &nbsp; &nbsp; &nbsp; | Description
--- | ---
[SageMaker training jobs](docs/sagemaker.md) | SageMaker users, we recommend you start with this page on how to run SageMaker training jobs with SageMaker Debugger
Framework Pages <ul><li>[TensorFlow](docs/tensorflow.md)</li><li>[PyTorch](docs/pytorch.md)</li><li>[MXNet](docs/mxnet.md)</li><li>[XGBoost](docs/xgboost.md)</li></ul> | See the frameworks pages for details on what's supported and how to modify your training script if applicable
[Programming Model for Analysis](docs/analysis.md) | For description of the programming model provided by our APIs which allows you to perform interactive exploration of tensors saved as well as to write your own Rules monitoring your training jobs.
[APIs](docs/api.md) | Full description of our APIs


## License
This library is licensed under the Apache 2.0 License.
