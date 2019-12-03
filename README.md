# Amazon SageMaker Debugger

- [Overview](#overview)
- [Examples](#sagemaker-example)
- [How It Works](#how-it-works)

## Overview
Amazon SageMaker Debugger is an AWS service to automatically debug your machine learning training process.
It helps you develop better, faster, cheaper models by catching common errors quickly. It supports
TensorFlow, PyTorch, MXNet, and XGBoost on Python 3.6+.

- Zero-code-change experience on SageMaker and AWS Deep Learning containers.
- Automated anomaly detection and state assertions.
- Realtime training job monitoring and visibility into any tensor value.
- Distributed training and TensorBoard support.

There are two ways to use it: Automatic mode and configurable mode.

- Automatic mode: No changes to your training script. Specify the rules you want and launch a SageMaker Estimator job.
- Configurable mode: More powerful, lets you specify exactly which tensors and collections to save. Use the Python API within your script.


## Example: Amazon SageMaker Zero-Code-Change
This example uses a zero-script-change experience, where you can use your training script as-is.
See the [example notebooks](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger) for more details.
```python
import sagemaker
from sagemaker.debugger import rule_configs, Rule, CollectionConfig

# Choose a built-in rule to monitor your training job
rule = Rule.sagemaker(
    rule_configs.exploding_tensor(),
    rule_parameters={
        "tensor_regex": ".*"
    },
    collections_to_save=[
        CollectionConfig(name="weights"),
        CollectionConfig(name="losses"),
    ],
)

# Pass the rule to the estimator
sagemaker_simple_estimator = sagemaker.tensorflow.TensorFlow(
    entry_point="script.py",
    role=sagemaker.get_execution_role(),
    framework_version="1.15",
    py_version="py3",
    rules=[rule],
)

sagemaker_simple_estimator.fit()
```

That's it! Amazon SageMaker will automatically monitor your training job for your and create a CloudWatch
event if you run into exploding tensor values.

If you want greater configuration and control, we offer that too. Simply


## Example: Running Locally
Requires Python 3.6+, and this example uses tf.keras. Run
```
pip install smdebug
```

To use Amazon SageMaker Debugger, simply add a callback hook:
```python
import smdebug.tensorflow as smd
hook = smd.KerasHook.(out_dir=args.out_dir)

model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)

# Add the hook as a callback
model.fit(x_train, y_train, epochs=args.epochs, callbacks=[hook])
model.evaluate(x_test, y_test, callbacks=[hook])

# Create a trial to inspect the saved tensors
trial = smd.create_trial(out_dir=args.out_dir)
print(f"Saved tensor values for {trial.tensors()}")
print(f"Loss values were {trial.tensor('CrossEntropyLoss:0')}")
```

## How It Works
Amazon SageMaker Debugger uses a `hook` to store the values of tensors throughout the training process. Another process called a `rule` job
simultaneously monitors and validates these outputs to ensure that training is progressing as expected.
A rule might check for vanishing gradients, or exploding tensor values, or poor weight initialization.
If a rule is triggered, it will raise a CloudWatch event, saving you time
and money.

Amazon SageMaker Debugger can be used inside or outside of SageMaker. There are three main use cases:
- SageMaker Zero-Script-Change: Here you specify which rules to use when setting up the estimator and run your existing script, no changes needed. See the first example above.
- SageMaker Bring-Your-Own-Container: Here you specify the rules to use, and modify your training script.
- Non-SageMaker: Here you write custom rules (or manually analyze the tensors) and modify your training script. See the second example above.

The reason for different setups is that SageMaker Zero-Script-Change (via Deep Learning Containers) uses custom framework forks of TensorFlow, PyTorch, MXNet, and XGBoost to save tensors automatically.
These framework forks are not available in custom containers or non-SM environments, so you must modify your training script in these environments.

See the [SageMaker page](docs/sagemaker.md) for details on SageMaker Zero-Code-Change and BYOC experience.\
See the frameworks pages for details on modifying the training script:
- [TensorFlow](docs/tensorflow.md)
- [PyTorch](docs/pytorch.md)
- [MXNet](docs/mxnet.md)
- [XGBoost](docs/xgboost.md)

## License

This library is licensed under the Apache 2.0 License.
