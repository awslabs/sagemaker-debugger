# Sagemaker Debugger

- [Overview](#overview)
- [SageMaker Example](#sagemaker-example)
- [Python Example](#python-example)
- [Concepts](#concepts)
- [Glossary](#glossary)
- [Detailed Links](#detailed-links)

## Overview
Sagemaker Debugger is an AWS service to automatically debug your machine learning training process.
It helps you develop better, faster, cheaper models by catching common errors quickly.

- Zero-code-change experience on SageMaker and AWS Deep Learning containers.
- Automated anomaly detection and state assertions.
- Realtime training job monitoring and visibility into any tensor value.
- Distributed training and TensorBoard support.

## SageMaker Example
This example uses a zero-code-change experience, where you can use your training script as-is.\
See the [sagemaker](https://link.com) page for more details.
```python
import sagemaker
from sagemaker.debugger import rule_configs, Rule, CollectionConfig

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

sagemaker_simple_estimator = sagemaker.tensorflow.TensorFlow(
    entry_point="script.py",
    role=sagemaker.get_execution_role(),
    framework_version="1.15",
    py_version="py3",
    rules=[rule],
)

sagemaker_simple_estimator.fit()
```


## Python Example
Requires Python 3.6+. Run
```
pip install smdebug
```

This example uses tf.keras. Say your training code looks like this:
```python
model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.fit(x_train, y_train, epochs=args.epochs)
model.evaluate(x_test, y_test)
```

To use Sagemaker Debugger, simply add a callback hook:
```python
import smdebug.tensorflow as smd
hook = smd.KerasHook(out_dir=args.out_dir)

model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.fit(x_train, y_train, epochs=args.epochs, callbacks=[hook])
model.evaluate(x_test, y_test, callbacks=[hook])
```

To analyze the result of the training run, create a trial and inspect the tensors.
```python
trial = smd.create_trial(out_dir=args.out_dir)
print(f"Saved tensor values for {trial.tensors()}")
print(f"Loss values were {trial.tensor('CrossEntropyLoss:0')}")
```

## Concepts
The steps to use Tornasole in any framework are:

1. Create a `hook`.
2. Register your model and optimizer with the hook.
3. Specify the `rule` to be used.
4. After training, create a `trial` to manually analyze the tensors.

Framework-specific details are here:
- [Tensorflow](https://link.com)
- [PyTorch](https://link.com)
- [MXNet](https://link.com)
- [XGBoost](https://link.com)

## Glossary

The imports assume `import smdebug.{tensorflow,pytorch,mxnet,xgboost} as smd`.

**Hook**: The main interface to use training. This object can be passed as a model hook/callback
in Tensorflow and Keras. It keeps track of collections and writes output files at each step.
- `hook = smd.Hook(out_dir="/tmp/mnist_job")`

**Mode**: One of "train", "eval", "predict", or "global". Helpful for segmenting data based on the phase
you're in. Defaults to "global".
- `train_mode = smd.modes.TRAIN`

**Collection**: A group of tensors. Each collection contains its own save configuration and regexes for
tensors to include/exclude.
- `collection = hook.get_collection("losses")`

**SaveConfig**: A Python dict specifying how often to save losses and tensors.
- `save_config = smd.SaveConfig(save_interval=10)`

**ReductionConfig**: Allows you to save a reduction, such as 'mean' or 'l1 norm', instead of the full tensor.
- `reduction_config = smd.ReductionConfig(reductions=['min', 'max', 'mean'], norms=['l1'])`

**Trial**: The main interface to use when analyzing a completed training job. Access collections and tensors. See [trials documentation](https://link.com).
- `trial = smd.create_trial(out_dir="/tmp/mnist_job")`

**Rule**: A condition that will trigger an exception and terminate the training job early, for example a vanishing gradient. See [rules documentation](https://link.com).

## Detailed Links
- [Rules and Trials](https://link.com)
- [Distributed Training](https://link.com)
- [TensorBoard](https://link.com)
