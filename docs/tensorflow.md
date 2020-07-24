# Tensorflow

## Contents
- [Support](#support)
- [How to Use](#how-to-use)
- [Code Structure Samples](#examples)
- [References](#references)

---

## Support

**Zero script change experience** — No modifications needed to your training script to enable the Debugger features while using the [official AWS Deep Learning Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html).

**Script mode experience** — The smdebug library supports training jobs with TensorFlow framework and script mode through its API operations. This option requires minimal changes to your training script.

### Versions
For a full list of TensorFlow framework versions to use Debugger, see [AWS Deep Learning Containers and SageMaker training containers](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html#debugger-supported-aws-containers).

### Distributed training supported by Debugger
- Horovod and Mirrored Strategy multi-GPU distributed trainings are supported.
- Parameter server based distributed training is currently not supported.

---

## How to Use
### Debugger with AWS Deep Learning Containers and zero script change

The Debugger features are all integrated into the AWS Deep Learning Containers, and you can run your training script with zero script change. To find a high-level SageMaker TensorFlow estimator with Debugger example code, see [Debugger in TensorFlow](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html#debugger-zero-script-change-TensorFlow).

### Debugger with AWS training containers and script Mode

In case you want to run your own training script and debug using the SageMaker TensorFlow framework with script mode and Debugger, the smdebug client library provides the hook constructor that you can add to the training script and retrieve tensors.

#### 1. Create a hook

 To create the hook constructor, add the following code.

```python
import smdebug.tensorflow as smd
hook = smd.{hook_class}.create_from_json_file()
```

Depending on the TensorFlow versions for your model, you need to choose a hook class. There are three hook constructor classes that you can pick and replace `{hook_class}`: `KerasHook`, `SessionHook`, and `EstimatorHook`.

#### KerasHook

Use if you use the Keras `model.fit()` API. This is available for all frameworks and versions of Keras and TensorFlow. `KerasHook` covers the eager execution modes and the gradient tape feature that are introduced from the TensorFlow framework version 2.0. For example, you can set the Keras hook constructor by adding the following code into your training script.
```python
hook = smd.KerasHook.create_from_json_file()
```
To learn how to fully implement the hook to your training script, see the [Keras with the TensorFlow gradient tape and the smdebug hook example scripts](https://github.com/awslabs/sagemaker-debugger/tree/master/examples/tensorflow2/scripts).

> **Note**: If you use the AWS Deep Learning Containers for zero script change, Debugger collects the most of tensors regardless the eager execution modes, through its high-level API.

#### SessionHook

Use if your model is created in TensorFlow version 1.x with the low-level approach, not using the Keras API. This is for the TensorFlow 1.x monitored training session API, `tf.train.MonitoredSessions()`.

```python
hook = smd.SessionHook.create_from_json_file()
```

To learn how to fully implement the hook into your training script, see the [TensorFlow monitored training session with the smdebug hook example script](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py).

> **Note**: The `tf.train.MonitoredSessions()` API is deprecated in favor of `tf.function()` in TF 2.0 and above. You can use `SessionHook` for `tf.function()` in TF 2.0 and above.

#### EstimatorHook

Use if you have a model using the `tf.estimator()` API. Available for any TensorFlow framework versions that supports the `tf.estimator()` API.

```python
hook = smd.EstimatorHook.create_from_json_file()
```

To learn how to fully implement the hook into your training script, see the [simple MNIST training script with the Tensorflow estimator](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py).
https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/local/mnist.py).

#### 2. Register the hook to your model

To collect the tensors from the hooks that you implemented, add `callbacks=[hook]` to the Keras `model.fit()` API and `hooks=[hook]` for the `MonitoredSession()`, `tf.function()`, and `tf.estimator()` APIs.

#### 3. Wrap the optimizer and the gradient tape

The smdebug TensorFlow hook provides tools to manually retrieve `gradients` tensors specific for the TensorFlow framework.

If you want to save `gradients` from the optimizer of your model, wrap it with the hook as follows:
```python
optimizer = hook.wrap_optimizer(optimizer)
```

If you want to save `gradients` from the TensorFlow gradient tape feature, wrap it as follows:
```python
with hook.wrap_tape(tf.GradientTape(persistent=True)) as tape:
```

These wrappers capture the gradient tensors, not affecting your optimization logic at all.

For examples of code structure to apply the hook wrappers, see [Examples](#examples)

#### 4. Take actions using the hook APIs

For a full list of actions that the hook APIs offer to construct hooks and save tensors, see [Common hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#common-hook-api) and [TensorFlow specific hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#tensorflow-specific-hook-api).

>**Note**: The `inputs`, `outputs`, and `layers` collections are not currently available for TensorFlow 2.1.

---

## Examples

The following examples show the three different hook constructions of TensorFlow. The following examples show what minimal changes have to be made to enable SageMaker Debugger while using the AWS containers with script mode. To learn how to use the high-level Debugger features with zero script change on AWS Deep Learning Containers, see [Use Debugger in AWS Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html).

### Keras API (tf.keras)
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

### Keras GradientTape example for TF 2.0 and above
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

### Monitored Session (tf.train.MonitoredSession)
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

### Estimator (tf.estimator.Estimator)
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

## References

### The smdebug API for saving tensors
See the [API for saving tensors](api.md) page for details about the Hooks, Collection, SaveConfig, and ReductionConfig.
See the [Analysis](analysis.md) page for details about analyzing a training job.

### TensorFlow References
- TF 1.x:
    - [tf.estimator](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/estimator)
    - [tf.keras](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras)
    - [tf.train.MonitoredSession](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/MonitoredSession?hl=en)
- TF 2.1:
    - [tf.estimator](https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/estimator)
    - [tf.keras](https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/keras)
- TF 2.2:
    - [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator)
    - [tf.keras](https://www.tensorflow.org/versions/r2.2/api_docs/python/tf)
