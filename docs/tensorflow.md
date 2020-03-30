# Tensorflow

## Contents
- [Support](#support)
- [How to Use](#how-to-use)
- [tf.keras Example](#tfkeras)
- [MonitoredSession Example](#monitoredsession)
- [Estimator Example](#estimator)
- [Full API](#full-api)

---

## Support

### Versions
- Zero Script Change experience where you need no modifications to your training script is supported in the official [SageMaker Framework Container for TensorFlow 1.15](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html), or the [AWS Deep Learning Container for TensorFlow 1.15](https://aws.amazon.com/machine-learning/containers/).

- This library itself supports the following versions when you use our API which requires a few minimal changes to your training script: TensorFlow 1.14, 1.15, 2.0.1, 2.1.0. Keras 2.3.

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

## Examples

We have three Hooks for different interfaces of TensorFlow. The following is needed to enable SageMaker Debugger on non Zero Script Change supported containers. Refer [SageMaker training](sagemaker.md) on how to use the Zero Script Change experience.

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

### TF 2.x GradientTape example
```python
import smdebug.tensorflow as smd
hook = smd.KerasHook(out_dir=args.out_dir)

model = tf.keras.models.Sequential([ ... ])
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            # wrap the tape to capture tensors
            with hook.wrap_tape(tf.GradientTape(persistent=True)) as tape:
                logits = model(data, training=True)  # (32,10)
                loss_value = cce(labels, logits)
            grads = tape.gradient(loss_value, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            acc = train_acc_metric(dataset_labels, logits)
            # manually save metric values
            hook.record_tensor_value(tensor_name="accuracy", tensor_value=acc)
```

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

---

## Full API
See the [API for saving tensors](api.md) page for details about the Hooks, Collection, SaveConfig, and ReductionConfig.
See the [Analysis](analysis.md) page for details about analyzing a training job.
