# Sagemaker Debugger

- [Overview](#overview)
- [Install](#install)
- [Example Usage](#example-usage)
- [Concepts](#concepts)

## Overview
Sagemaker Debugger is an AWS service to automatically debug your machine learning training process.
It helps you develop better, faster, cheaper models by catching common errors quickly.

## Install
```
pip install smdebug
```

Requires Python 3.6+.

## Example Usage
This example uses Keras. Say your training code looks like this:
```
model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.fit(x_train, y_train, epochs=args.epochs)
model.evaluate(x_test, y_test)
```

To use Sagemaker Debugger, simply add a callback hook:
```
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
```
trial = smd.create_trial(out_dir=args.out_dir)
print(f"Saved tensor values for {trial.tensors()}")
print(f"Loss values were {trial.get_collection("losses").values()}")
```


## Concepts
The steps to use Tornasole in any framework are:

1. Create a `hook`.
2. Register your model and optimizer with the hook.
3. Specify the `rule` to be used.
4. After training, create a `trial` to manually analyze the tensors.

See the [glossary](https://link.com) to understand these terms better.

Framework-specific details are here:
- [Tensorflow](https://link.com)
- [PyTorch](https://link.com)
- [MXNet](https://link.com)
- [XGBoost](https://link.com)
