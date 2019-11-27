# Tensorflow

SageMaker Zero-Code-Change supported container: TensorFlow 1.15. See the [AWS Docs](https://link.com) for details.\
Python API supported versions: Tensorflow 1.13, 1.14, 1.15. Keras 2.3.

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
Adds functionality to the optimizer object to log gradients. Returns the original optimizer and doesn't change the optimization process.

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
Adds functionality to the optimizer object to log gradients. Returns the original optimizer and doesn't change the optimization process.

## Concepts
The steps to use Tornasole in any framework are:

1. Create a `hook`.
2. Register your model and optimizer with the hook.
3. Specify the `rule` to be used.
4. After training, create a `trial` to manually analyze the tensors.

See the [API page](https://link.com) for more details.

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
- [Full API](https://link.com)
- [Rules and Trials](https://link.com)
- [Distributed Training](https://link.com)
- [TensorBoard](https://link.com)
