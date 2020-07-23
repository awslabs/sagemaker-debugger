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
* Zero Script Change experience — No modifications needed to your training script to enable the Debugger features while using the official AWS Deep Learning Containers.
* Script mode experience — This smdebug library supports training jobs with TensorFlow framework and script mode through its API operations, which require a minimal changes to your training script.
* For a full list of TensorFlow framework versions to use Debugger, see [AWS Deep Learning Containers and SageMaker training containers](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html#debugger-supported-aws-containers).


### Distributed training supported by Debugger
- Horovod and Mirrored Strategy multi-GPU distributed trainings are supported.
- Parameter server based distributed training is currently not supported.


## How to Use
### Debugger with AWS Deep Learning Containers and Zero Script Change

The Debugger features are all integrated into the AWS Deep Learning Containers, and you can run your training script with zero script change. To find a high-level SageMaker TensorFlow estimator with Debugger example code, see [Debugger in TensorFlow](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html#debugger-zero-script-change-TensorFlow).

### Debugger with AWS Training Containers and Script Mode

### 1. Create a hook

If you want to use your own training script using the SageMaker TensorFlow framework with script mode and Debugger, the smdebug client library provides the hook constructor that you can add to the training script and retrieve tensors. To create the hook constructor, add the following code.

```
`import smdebug.tensorflow as smd
hook = smd.{hook_class}.create_from_json_file()`
```

Depending on the TensorFlow versions for your model, you need to choose a hook class. There are three hook constructor classes that you can pick and replace `{hook_class}`.

* `KerasHook` — Use if you are using the model.fit() API of Keras with TensorFlow backend. This is available for all Keras TensorFlow framework versions. KerasHook covers the eager execution modes that is introduced from TensorFlow  version 2.0. The `hook.wrap_tape(tf.GradientTape())` collects the tensors from the TensorFlow gradient tape function.
    * example: https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf_keras_gradienttape.py
    * Note: If you use the AWS Deep Learning Containers for zero script change, Debugger collects the most of tensors regardless the eager execution modes, through its high-level API.
* `SessionHook` — Use if your model was created in TensorFlow version 1.x with a low-level approach, not using the keras API. This is for the TensorFlow 1.x monitored training session API, `tf.train.MonitoredSessions()`.
    * example: https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py
    * Note: the `tf.train.MonitoredSessions()` API is deprecated in favor of favor of `tf.function()` in TF 2.0 and above.
* `EstimatorHook` — Use if you have a model using the `tf.estimator()` API. Available for TensorFlow framework versions that supports the `tf.estimator()` API.

### 2. Register the hook to your model

To collect the tensors from the hooks that you implemented, add `callbacks=[hook]` to the Keras TensorFlow `model.fit()` API and `hooks=[hook]` for the `MonitoredSession()`, `tf.function()`, and `tf.estimator()` APIs.

### 3. Wrap the optimizer

If you would like to save `gradients` from the optimizer of your model, wrap it with the hook as follows `optimizer = hook.wrap_optimizer(optimizer)`. This does not modify your optimization logic and returns the same optimizer instance passed to the method.

### 4. (Optional) Configure Collections, SaveConfig and ReductionConfig

See the [Common API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md) page for details on how to do this.


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

## References
- TF 1.x:
    - [Estimator](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/estimator)
    - [tf.keras](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras)
    - [MonitoredSession](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/MonitoredSession?hl=en)
- TF 2.1:
    - [Estimator](https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/estimator)
    - [tf.keras](https://www.tensorflow.org/versions/r2.1/api_docs/python/tf/keras)
- TF 2.2:
    - [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator)
    - [tf.keras](https://www.tensorflow.org/versions/r2.2/api_docs/python/tf)
