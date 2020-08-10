# Tensorflow

## Contents
- [What SageMaker Debugger Supports](#support)
- [How to Use Debugger with TensorFlow](#how-to-use)
  - [Debugger on AWS Deep Learning Containers with TensorFlow](#debugger-dlc)
  - [Debugger on SageMaker Training Containers and Custom Containers](#debugger-script-change)
- [Code Samples](#examples)
- [References](#references)

---

## What SageMaker Debugger Supports <a name="support"></a>

SageMaker Debugger python SDK (v2.0) and its client library `smdebug` library (v0.9.1) now fully support TensorFlow 2.2 with the latest version release. Using Debugger, you can access tensors of any kind of TensorFlow models, from the Keras model zoo to your custom model, and save them using Debugger built-in or custom tensor collections.
You can simply run your training script on [the official AWS Deep Learning Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html) where Debugger can automatically capture tensors from your training job. No matter what your TensorFlow models use Keras API or pure TensorFlow API, in eager mode or non-eager mode, you can directly run them on the AWS Deep Learning Containers.  

Debugger and its client library `smdebug` support debugging your training job on other AWS training containers and custom containers. In this case, a hook registration process is required to manually add the hook features to your training script. For a full list of AWS TensorFlow containers to use Debugger, see [SageMaker containers to use Debugger with script mode](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html#debugger-supported-aws-containers). For a complete guide of using custom containers, go to [Use Debugger in Custom Training Containers ](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-bring-your-own-container.html).

### New features
- The latest TensorFlow version fully covered by Debugger is `2.2.0`.
- Debug training jobs with the TensorFlow framework or Keras TensorFlow.
- Debug training jobs with the TensorFlow eager or non-eager mode.
- New built-in tensor collections: `inputs`, `outputs`, `layers`, `gradients`.
- New hook APIs to save tensors, in addition to scalars: `save_tensors`, `save_scalar`.

### Distributed training supported by Debugger
- Horovod and Mirrored Strategy multi-GPU distributed trainings are supported.
- Parameter server based distributed training is currently not supported.

---

## How to Use Debugger <a name="how-to-use"></a>

### Debugger on AWS Deep Learning Containers with TensorFlow <a name="debugger-dlc"></a>

The Debugger built-in rules and hook features are fully integrated into the AWS Deep Learning Containers, and you can run your training script without any script changes. When running training jobs on those Deep Learning Containers, Debugger registers its hooks automatically to your training script in order to retrieve tensors. To find a comprehensive guide of using the high-level SageMaker TensorFlow estimator with Debugger, go to the [Amazon SageMaker Debugger with TensorFlow](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html#debugger-zero-script-change-TensorFlow) developer guide.

The following code sample is the base structure of a SageMaker TensorFlow estimator with Debugger.

```python
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import Rule, DebuggerHookConfig, CollectionConfig, rule_configs

tf_estimator = TensorFlow(
    entry_point = "tf-train.py",
    role = "SageMakerRole",
    instance_count = 1,
    instance_type = "ml.p2.xlarge",
    framework_version = "2.2.0",
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

>**Note**: The SageMaker TensorFlow estimator and the Debugger collections in this example are based on the latest SageMaker Python SDK v2.0 and `smdebug` v0.9.1. It is highly recommended to upgrade the packages by executing the following command lines.
    ```
    pip install -U sagemaker
    pip install -U smdebug
    ```
>If you are using Jupyter Notebook, put exclamation mark at the front of the code lines and restart your kernel. For more information about breaking changes of the SageMaker Python SDK, see [Use Version 2.x of the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/v2.html).

#### Available Tensor Collections for TensorFlow

The following table lists the pre-configured tensor collections for TensorFlow models. You can pick any tensor collections by specifying the `name` parameter of `CollectionConfig()` as shown in the base code sample.

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

For more information about adjusting the tensor collection parameters, see [Save Tensors Using Debugger Modified Built-in Collections ](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-data.html#debugger-save-modified-built-in-collections).

For a full list of available tensor collection parameters, see [Configuring Collection using SageMaker Python SDK](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#configuring-collection-using-sagemaker-python-sdk).

>**Note**: The `inputs`, `outputs`, and `layers` collections are not currently available for TensorFlow 2.1.

### Debugger on SageMaker Training Containers and Custom Containers <a name="debugger-script-change"></a>

If you want to run your own training script or custom containers other than the AWS Deep Learning Containers in the previous option, there are two alternatives.
- Alternative 1: Use the SageMaker TensorFlow training containers with training script modification
- Alternative 2: Use your custom container with modified training script and push the container to Amazon ECR.
In both cases, you need to manually register the Debugger hook to your training script. Depending on the TensorFlow and Keras API operations used to construct your model, you need to pick the right TensorFlow hook class, register the hook, and save tensors.

1. [Create a hook](#create-a-hook)
    - [KerasHook](#kerashook)
    - [SessionHook](#sessionhook)
    - [EstimatorHook](#estimatorhook)
2. [Wrap the optimizer and the gradient tape with the hook to retrieve gradient tensors](#wrap-opt-with-hook)
3. [Register the hook to model.fit()](#register-a-hook)


#### 1. Create a hook <a name="create-a-hook"></a>

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

If you want to save gradients and outputs tensors from the TensorFlow `GradientTape` feature, wrap `tf.GradientTape` with the smdebug `hook.wrap_tape` method and save using the `hook.save_tensor` function. The input of `hook.save_tensor` is in (tensor_name, tensor_value, collections_to_write="default") format. For example:
```python
with hook.wrap_tape(tf.GradientTape(persistent=True)) as tape:
    logits = model(data, training=True)
    loss_value = cce(labels, logits)
hook.save_tensor("y_labels", labels, "outputs")
hook.save_tensor("predictions", logits, "outputs")
grads = tape.gradient(loss_value, model.variables)
hook.save_tensor("grads", grads, "gradients")
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

## Code Samples <a name="examples"></a>

The following examples show the base structures of hook registration in various TensorFlow training scripts. If you want to take the benefit of the high-level Debugger features with zero script change on AWS Deep Learning Containers, see [Use Debugger in AWS Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html).

### Keras API (tf.keras)
```python
import smdebug.tensorflow as smd

hook = smd.KerasHook.create_from_json_file()

model = tf.keras.models.Sequential([ ... ])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
# Add the hook as a callback
hook.set_mode(mode=smd.modes.TRAIN)
model.fit(x_train, y_train, epochs=args.epochs, callbacks=[hook])

hook.set_mode(mode=smd.modes.EVAL)
model.evaluate(x_test, y_test, callbacks=[hook])
```

### Keras GradientTape example for TF 2.0 and above
```python
import smdebug.tensorflow as smd

hook = smd.KerasHook.create_from_json_file()

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
            hook.save_tensor(tensor_name="accuracy", tensor_value=acc, collections_to_write="default")
```

### Monitored Session (tf.train.MonitoredSession)
```python
import smdebug.tensorflow as smd

hook = smd.SessionHook.create_from_json_file()

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

hook = smd.EstimatorHook.create_from_json_file()

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
