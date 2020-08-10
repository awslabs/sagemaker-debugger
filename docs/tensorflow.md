# Tensorflow

## Contents
- [Support](#support)
- [How to Use Debugger with TensorFlow](#how-to-use)
  - [Debugger with AWS Deep Learning Containers](#debugger-dlc)
  - [Debugger with other AWS training containers and custom containers](#debugger-script-change)
- [Code Structure Samples](#examples)
- [References](#references)

---

## Support

### Supported TensorFlow Versions

The SageMaker Debugger python SDK and `smdebug` library now fully support TensorFlow 2.2 with the latest version release. Using Debugger, you can retrieve tensors from your TensorFlow models with either eager or non-eager mode, with Keras API or the pure TensorFlow framework.
For a full list of TensorFlow framework versions to use Debugger, see [AWS Deep Learning Containers and SageMaker training containers](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html#debugger-supported-aws-containers).

**Zero script change experience** — No modification is needed to your training script to enable the Debugger features while using the [official AWS Deep Learning Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html).

**Script mode experience** — The smdebug library supports training jobs with the TensorFlow framework and script mode through its API operations. This option requires minimal changes to your training script to register Debugger hooks, and the smdebug library provides you hook features to help implement Debugger and analyze saved tensors.

### Distributed training supported by Debugger
- Horovod and Mirrored Strategy multi-GPU distributed trainings are supported.
- Parameter server based distributed training is currently not supported.

---

## How to Use Debugger

### Debugger with AWS Deep Learning Containers <a name="debugger-dlc"></a>

The Debugger built-in rules and hook features are fully integrated into the AWS Deep Learning Containers, and you can run your training script without any script changes. When running training jobs on those Deep Learning Containers, Debugger registers its hooks automatically to your training script in order to retrieve tensors. To find a comprehensive guide of using the high-level SageMaker TensorFlow estimator with Debugger, see [Debugger in TensorFlow](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html#debugger-zero-script-change-TensorFlow).

The following code sample is how to set a SageMaker TensorFlow estimator with Debugger.

```python
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import Rule, DebuggerHookConfig, CollectionConfig, rule_configs

tf_estimator = TensorFlow(
    entry_point = "tf-train.py",
    role = "SageMakerRole",
    instance_count = 1,
    instance_type = "ml.p2.xlarge",
    framework_version = "2.2",
    py_version = "py37"

    # Debugger-specific Parameters
    rules = [
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
        ...
    ],
    debugger_hook_config = DebuggerHookConfig(
        CollectionConfig(name="inputs"),
        CollectionConfig(name="outputs"),
        CollectionConfig(name="layers"),
        CollectionConfig(name="gradients")
        ...
    )
)
tf_estimator.fit("s3://bucket/path/to/training/data")
```

Available tensor collections that you can retrieve from TensorFlow training jobs for zero script change are as follows:

| Name | Description|
| --- | --- |
| all	| Matches all tensors. |
| default |	Includes "metrics", "losses", and "sm_metrics". |
| metrics |	For KerasHook, saves the metrics computed by Keras for the model. |
| losses | Saves all losses of the model. |
| sm_metrics | You can add scalars that you want to show up in SageMaker Metrics to this collection. SageMaker Debugger will save these scalars both to the out_dir of the hook, as well as to SageMaker Metric. Note that the scalars passed here will be saved on AWS servers outside of your AWS account. |
| inputs | Matches all input to the model. |
| outputs |	Matches all outputs of the model, such as predictions (logits) and labels. |
| layers | Matches all inputs and outputs of intermediate layers. |
| gradients |	Matches all gradients of the model. In TensorFlow when not using zero script change environments, must use hook.wrap_optimizer() or hook.wrap_tape(). |
| weights |	Matches all weights of the model. |
| biases |	Matches all biases of the model. |
| optimizer_variables |	Matches all optimizer variables, currently only supported for Keras. |

>**Note**: The `inputs`, `outputs`, and `layers` collections are not currently available for TensorFlow 2.1.

### Debugger with other AWS training containers and custom containers <a name="debugger-script-change"></a>

If you want to run your own training script or custom container, there are two available options. One option is to use the SageMaker TensorFlow with script change on other AWS training containers (the SageMaker TensorFlow estimator is in script mode by default from TensorFlow 2.1, so you do not need to specify `script_mode` parameter). Another option is to use your custom container with your training script and push the container to Amazon ECR. In both cases, you need to manually register the Debugger hook to your training script. Depending on the TensorFlow models and API operations in your script, you need to pick the right hook class as introduced in the following steps.

1. [Create a hook](#create-a-hook)
  * [KerasHook](#kerashook)
  * [SessionHook](#sessionhook)
  * [EstimatorHook](#estimatorhook)
2. [Wrap the optimizer and the gradient tape with the hook to retrieve gradient tensors](#wrap-opt-with-hook)
3. [Register the hook to model.fit()](#register-a-hook)


#### 1. Create a hook

 To create the hook constructor, add the following code to your training script. This will enable the `smdebug` tools for TensorFlow and create a TensorFlow hook object.

```python
import smdebug.tensorflow as smd
hook = smd.{hook_class}.create_from_json_file()
```

Depending on TensorFlow versions and Keras API that was used in your training script, you need to choose the right hook class. There are three hook constructors for TensorFlow that you can choose: `KerasHook`, `SessionHook`, and `EstimatorHook`.

#### KerasHook

Use `KerasHook` if you use the Keras model zoo and a Keras `model.fit()` API. This is available for the Keras with TensorFlow backend interface. `KerasHook` covers the eager execution modes and the gradient tape features that are introduced from the TensorFlow framework version 2.0. You can set the smdebug Keras hook constructor by adding the following code into your training script. Place this code line before `model.compile()`.

```python
hook = smd.KerasHook.create_from_json_file()
```

To learn how to fully implement the hook to your training script, see the [Keras with the TensorFlow gradient tape and the smdebug hook example scripts](https://github.com/awslabs/sagemaker-debugger/tree/master/examples/tensorflow2/scripts).

>**Note**: If you use the AWS Deep Learning Containers for zero script change, Debugger collects the most of tensors regardless the eager execution modes, through its high-level API.

#### SessionHook

Use if your model is created in TensorFlow version 1.x with the low-level approach, not using the Keras API. This is for the TensorFlow 1.x monitored training session API, `tf.train.MonitoredSessions()`.

```python
hook = smd.SessionHook.create_from_json_file()
```

To learn how to fully implement the hook into your training script, see the [TensorFlow monitored training session with the smdebug hook example script](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py).

>**Note**: The official TensorFlow library deprecated the `tf.train.MonitoredSessions()` API in favor of `tf.function()` in TF 2.0 and above. You can use `SessionHook` for `tf.function()` in TF 2.0 and above.

#### EstimatorHook

Use if you have a model using the `tf.estimator()` API. Available for any TensorFlow framework versions that supports the `tf.estimator()` API.

```python
hook = smd.EstimatorHook.create_from_json_file()
```

To learn how to fully implement the hook into your training script, see the [simple MNIST training script with the Tensorflow estimator](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/sagemaker_byoc/simple.py).

#### 2. Wrap the optimizer and the gradient tape to retrieve gradient tensors <a name="wrap-opt-with-hook"></a>

The smdebug TensorFlow hook provides tools to manually retrieve `gradients` tensors specific for the TensorFlow framework.

If you want to save `gradients`, for example, from the Keras Adam optimizer, wrap it with the hook as follows:
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
optimizer = hook.wrap_optimizer(optimizer)
```

If you want to save `gradients` from the TensorFlow gradient tape feature, wrap `tf.GradientTape` with the `hook.wrap_tape` method and save using the `hook.save_tensor` function. The input of `hook.save_tensor` is in (tensor_name, tensor_value, collections_to_write="default") format. For example:
```python
with hook.wrap_tape(tf.GradientTape(persistent=True)) as tape:
    logits = model(data, training=True)
    loss_value = cce(labels, logits)
hook.save_tensor("y_labels", labels, "outputs")
grads = tape.gradient(loss_value, model.variables)
```

These smdebug hook wrapper functions capture the gradient tensors, not affecting your optimization logic at all.

For examples of code structure to apply the hook wrappers, see the [Examples](#examples) section.

#### 3. Register the hook to model.fit() <a name="register-a-hook"></a>

To collect the tensors from the hooks that you registered, add `callbacks=[hook]` to the Keras `model.fit()` API. This will pass the SageMaker Debugger hook as a Keras callback. Similarly, add `hooks=[hook]` to the `MonitoredSession()`, `tf.function()`, and `tf.estimator()` APIs. For example:

```python
model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epoch,
              validation_data=(X_valid, Y_valid),
              shuffle=True,
              # smdebug modification: Pass the hook as a Keras callback
              callbacks=[hook])
```

#### 4. Take actions using the hook APIs

For a full list of actions that the hook APIs offer to construct hooks and save tensors, see [Common hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#common-hook-api) and [TensorFlow specific hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#tensorflow-specific-hook-api).

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
