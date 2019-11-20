# Tensorflow

Supported Tensorflow versions: 1.13, 1.14, and 1.15

- [Keras Example](#keras-example)
- [MonitoredSession Example](#monitored-session-example)
- [Estimator Example](#estimator-example)
- [Full API](#full-api)

## Keras Example
```
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
```
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
```
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

## KerasHook
```
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
)
```
Initializes the hook. Pass this object as a callback to Keras' `model.fit(), model.evaluate(), model.evaluate()`.

`out_dir` (str): Where to write the recorded tensors and metadata\
`export_tensorboard` (bool): Whether to use TensorBoard logs\
`tensorboard_dir` (str): Where to save TensorBoard logs\
`dry_run` (bool): If true, don't write any files\
`reduction_config` (list[str]): See glossary\
`save_config` (SaveConfig object): TODO\
`include_regex` (list[str]): List of additional regexes to save\
`include_collections` (list[str]): List of collections to save\
`save_all` (bool): Remove this???


```
wrap_optimizer(
    self,
    optimizer: Tuple[tf.train.Optimizer, tf.keras.Optimizer],
)
```
Adds callback methods to the optimizer object and returns it.

`optimizer` (Tuple[tf.train.Optimizer, tf.keras.Optimizer]): The optimizer


## EstimatorHook / SessionHook
These are aliases pointing to the same object.
```

__init__(
    out_dir: str,
    export_tensorboard: bool = False,
    tensorboard_dir: str = None,
    dry_run: bool = False,
    reduction_config: list[str] = None,
    save_config: smdebug.tensorflow.save_config = None,
    include_regex: list[str] = None,
    include_collections: list[str] = None,
    save_all = False,
    include_workers = "one"
)
```

See all parameters from `KerasHook`. Pass this object as a hook to tf.train.MonitoredSession's `run()` method.

`include_workers` (str): Used for distributed training, can also be "all".

```
# Adds callback methods to the optimizer object and returns it.
wrap_optimizer(
    self,
    optimizer: tf.train.Optimizer,
)
```
Adds callback methods to the optimizer object and returns it.
