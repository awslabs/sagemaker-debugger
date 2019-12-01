# Tensorflow

SageMaker Zero-Code-Change supported container: TensorFlow 1.15. See the [AWS Docs](https://link.com) for details.\
Python API supported versions: Tensorflow 1.13, 1.14, 1.15. Keras 2.3.



## Contents
- [How to Use](#how-to-use)
- [Keras Example](#keras-example)
- [MonitoredSession Example](#monitored-session-example)
- [Estimator Example](#estimator-example)
- [Full API](#full-api)

---

## How to Use
```import smdebug.tensorflow as smd```
#### 1. Create a hook
If using SageMaker, you will configure the hook in SageMaker Estimator. Instantiate it with
`smd.{hook_class}.create_from_json_file()`.\
Otherwise, call the hook class constructor, `smd.{hook_class}()`. Details are below for tf.keras, MonitoredSession, or Estimator.

#### 2. Pass the hook to the model as a callback
The keyword is `callbacks=[hook]` for tf.keras. It is `hooks=[hook]` for MonitoredSession and Estimator.

#### 3. (Optional) Wrap the optimizer
If you are accessing the GRADIENTS collection, and you are in BYOC or non-SageMaker mode, call `optimizer = hook.wrap_optimizer(optimizer)`.

#### 4. (Optional) Configure collections, SaveConfig, ReductionConfig
See the [Common API](https://link.com) page for details on how to do this.

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

Use this if in a non-SageMaker environment. In SageMaker, call `smd.KerasHook.create_from_json_file()`.
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
    optimizer: Union[tf.train.Optimizer, tf.keras.Optimizer],
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
Use this if in a non-SageMaker environment. In SageMaker, call `smd.SessionHook.create_from_json_file()`.

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
Use this if in a non-SageMaker environment. In SageMaker, call `smd.EstimatorHook.create_from_json_file()`.

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

---
See the [Common API](https://link.com) page for details about Collection, SaveConfig, and ReductionConfig.\
See the [Analysis](https://link.com) page for details about analyzing a training job.
