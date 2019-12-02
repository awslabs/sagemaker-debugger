# Tensorflow

## Contents
- [Support](#support)
- [How to Use](#how-to-use)
- [Keras Example](#keras-example)
- [MonitoredSession Example](#monitored-session-example)
- [Estimator Example](#estimator-example)
- [Full API](#full-api)
---
## Support

### Versions
- Zero Script Change experience where you need no modifications to your training script is supported in the official [SageMaker Framework Container for TensorFlow 1.15](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html), or the [AWS Deep Learning Container for TensorFlow 1.15](https://aws.amazon.com/machine-learning/containers/).

- This library itself supports the following versions when you use our API which requires a few minimal changes to your training script: TensorFlow 1.13, 1.14, 1.15. Keras 2.3.

### Interfaces
- [Estimator](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/estimator)
- [tf.keras](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras)
- [MonitoredSession](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/MonitoredSession?hl=en)

### Distributed training
- [MirroredStrategy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/distribute/MirroredStrategy) or [Contrib MirroredStrategy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/distribute/MirroredStrategy)

We will very quickly follow up with support for Horovod and Parameter Server based training.

---

## How to Use
### Using Zero Script Change containers
In this case, you don't need to do anything to get the hook running. You are encouraged to configure the hook from the SageMaker python SDK so you can run different jobs with different configurations without having to modify your script. If you want access to the hook to configure certain things which can not be configured through the SageMaker SDK, you can retrieve the hook as follows.
```
import smdebug.tensorflow as smd
hook = smd.{hook_class}.create_from_json_file()
```
Note that you can create the hook from smdebug's python API as is being done in the next section even in such containers.

### Bring your own container experience
#### 1. Create a hook
If using SageMaker, you will configure the hook in SageMaker's python SDK using the Estimator class. Instantiate it with
`smd.{hook_class}.create_from_json_file()`. Otherwise, call the hook class constructor, `smd.{hook_class}()`. Details are below for tf.keras, MonitoredSession, or Estimator.

#### 2. Register the hook to your model
The argument is `callbacks=[hook]` for tf.keras. It is `hooks=[hook]` for MonitoredSession and Estimator.

#### 3. Wrap the optimizer
If you would like to save `gradients`, wrap your optimizer with the hook as follows `optimizer = hook.wrap_optimizer(optimizer)`. This does not modify your optimization logic, and returns the same optimizer instance passed to the method.

#### 4. (Optional) Configure Collections, SaveConfig and ReductionConfig
See the [Common API](api.md) page for details on how to do this.

---

## tf.keras
### Example
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

### KerasHook
In SageMaker, call `smd.KerasHook.create_from_json_file()`.

In a non-SageMaker environment, use the following constructor.
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
    optimizer: Union[tf.train.Optimizer, tf.keras.Optimizer]
)
```
Adds functionality to the optimizer object to log gradients. Returns the original optimizer and doesn't change the optimization process.

`optimizer` (Union[tf.train.Optimizer, tf.keras.Optimizer]): The optimizer.

---

## MonitoredSession
### Example
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

### SessionHook
In SageMaker, call `smd.SessionHook.create_from_json_file()`.

If in a non-SageMaker environment, use the following constructor.

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
    optimizer: tf.train.Optimizer
)
```
Adds functionality to the optimizer object to log gradients. Returns the original optimizer and doesn't change the optimization process.

---

## Estimator
### Example
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

### EstimatorHook
In SageMaker, call `smd.EstimatorHook.create_from_json_file()`.

If in a non-SageMaker environment, use the following constructor.

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
    optimizer: tf.train.Optimizer
)
```
Adds functionality to the optimizer object to log gradients. Returns the original optimizer and doesn't change the optimization process.

---
See the [Common API](api.md) page for details about Collection, SaveConfig, and ReductionConfig.\
See the [Analysis](analysis.md) page for details about analyzing a training job.
