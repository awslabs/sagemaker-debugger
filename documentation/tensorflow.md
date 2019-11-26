# Tensorflow

Supported Tensorflow versions: 1.13, 1.14, and 1.15.
Supported standalone Keras version: 2.3.

## Contents
- [Keras Example](#keras-example)
- [MonitoredSession Example](#monitored-session-example)
- [Estimator Example](#estimator-example)
- [Full API](#full-api)

## tf.keras Example
```python
import smdebug.tensorflow as smd
hook = smd.KerasHook(out_dir=args.out_dir)

model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
# Add the hook as a callback
model.fit(x_train, y_train, epochs=args.epochs, callbacks=[hook])
model.evaluate(x_test, y_test, callbacks=[hook])
```

## MonitoredSession Example
```python
import smdebug.tensorflow as smd
hook = smd.SessionHook(out_dir=args.out_dir)

loss = tf.reduce_mean(tf.matmul(...), name="loss")
optimizer = tf.train.AdamOptimizer(args.lr)

# Wrap the optimizer
optimizer = hook.wrap_optimizer(optimizer)

# Add the hook as a callback
sess = tf.train.MonitoredSession(hooks=[hook])

sess.run([loss, ...])
```

## Estimator Example
```python
import smdebug.tensorflow as smd
hook = smd.EstimatorHook(out_dir=args.out_dir)

train_input_fn, eval_input_fn = ...
estimator = tf.estimator.Estimator(...)

# Set the mode and pass the hook as callback
hook.set_mode(mode=smd.modes.TRAIN)
estimator.train(input_fn=train_input_fn, steps=args.steps, hooks=[hook])

hook.set_mode(mode=smd.modes.EVAL)
estimator.evaluate(input_fn=eval_input_fn, steps=args.steps, hooks=[hook])
```


# Full API

See the [Common API](https://link.com) page for details about Collection, SaveConfig, and ReductionConfig.\
See the [Analysis](https://link.com) page for details about analyzing a training job.

## KerasHook
```python
__init__(
    out_dir,
    export_tensorboard = False,
    tensorboard_dir = None,
    dry_run = False,
    reduction_config = None,
    save_config = None,
    include_regex = None,
    include_collections = None,
    save_all = False,
)
```
Initializes the hook. Pass this object as a callback to Keras' `model.fit(), model.evaluate(), model.evaluate()`.

`out_dir` (str): Where to write the recorded tensors and metadata.\
`export_tensorboard` (bool): Whether to use TensorBoard logs.\
`tensorboard_dir` (str): Where to save TensorBoard logs.\
`dry_run` (bool): If true, don't write any files.\
`reduction_config` (ReductionConfig object): See the Common API page.\
`save_config` (SaveConfig object): See the Common API page.\
`include_regex` (list[str]): List of additional regexes to save.\
`include_collections` (list[str]): List of collections to save.\
`save_all` (bool): Saves all tensors and collections. May be memory-intensive and slow.


```python
wrap_optimizer(
    self,
    optimizer: Tuple[tf.train.Optimizer, tf.keras.Optimizer],
)
```
Modify the optimizer object to log gradients, and return the optimizer. Must be used when saving gradients.

`optimizer` (Tuple[tf.train.Optimizer, tf.keras.Optimizer]): The optimizer.


## EstimatorHook / SessionHook
EstimatorHook is used for the tf.estimator.Estimator interface.\
SessionHook is used for tf.train.MonitoredSession objects (tf.Session objects are not supported).\
Because Estimator uses MonitoredSession under the hood, these names are aliases to the same class. They have two separate names for clarity.
```python
__init__(
    out_dir,
    export_tensorboard = False,
    tensorboard_dir = None,
    dry_run = False,
    reduction_config = None,
    save_config = None,
    include_regex = None,
    include_collections= None,
    save_all = False,
    include_workers = "one"
)
```

Pass this object as a hook to tf.train.MonitoredSession's `run()` method.

`out_dir` (str): Where to write the recorded tensors and metadata.\
`export_tensorboard` (bool): Whether to use TensorBoard logs.\
`tensorboard_dir` (str): Where to save TensorBoard logs.\
`dry_run` (bool): If true, don't write any files.\
`reduction_config` (ReductionConfig object): See the Common API page.\
`save_config` (SaveConfig object): See the Common API page.\
`include_regex` (list[str]): List of additional regexes to save.\
`include_collections` (list[str]): List of collections to save.\
`save_all` (bool): Saves all tensors and collections. May be memory-intensive and slow.\
`include_workers` (str): Used for distributed training, can also be "all".

```python
wrap_optimizer(
    self,
    optimizer: tf.train.Optimizer,
)
```
Modify the optimizer object to log gradients, and return the optimizer. Must be used when saving gradients.
