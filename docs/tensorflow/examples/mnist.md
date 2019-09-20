# MNIST Example
We provide an example script `mnist.py` which is a Tornasole-enabled TensorFlow training script.
It uses the Estimator interface of TensorFlow.
In this document we highlight how you can set training and evaluation modes for Tornasole.
This will allow you to distinguish between training and evaluate steps and analyze them independently.

## Integrating Tornasole
Below we call out the changes for Tornasole in the above script and describe them

**Importing TornasoleTF**
```
import tornasole.tensorflow as ts
```
**Saving gradients**

We need to wrap our optimizer with TornasoleOptimizer, and use this optimizer to minimize loss.
This will also enable us to access the gradients during analysis without having to identify which tensors out of the saved ones are the gradients.
```
opt = TornasoleOptimizer(opt)
optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)
```
Note that here since by default Tornasole tries to save weights, gradients and losses
we didn't need to specify 'gradients' in the include_collections argument of the hook.

**Saving losses**

Since we use a default loss function from Tensorflow here,
we would only need to indicate to the hook that we want to include losses.
But since the hook by default saves losses if include_collections argument was not set,
we need not do anything.

**Setting save interval**

You can set different save intervals for different modes.
This can be done by passing a dictionary as save_config to the hook.
This dictionary should have the mode as key and a SaveConfig object as value.
```
ts.TornasoleHook(...,
    save_config={ts.modes.TRAIN: ts.SaveConfig(args.tornasole_train_frequency),
                 ts.modes.EVAL: ts.SaveConfig(args.tornasole_eval_frequency)}..)
```
**Setting the right mode**

Notice the calls to `hook.set_mode` at various places in the code.
```
hook.set_mode(ts.modes.TRAIN)
```

```
hook.set_mode(ts.modes.EVAL)
```
**Passing the hook**

We need to pass this hook to a monitored session and use this session for running the job.
```
hook = ts.TornasoleHook(...)
mnist_classifier.train(..., hooks=[hook])
```

```
mnist_classifier.evaluate(..., hooks=[hook])
```
## Running the example
### Environment
Ensure you are in a python environment which has TensorFlow, TornasoleTF and TornasoleCore installed. If you followed the recommended instructions of using Amazon Deep Learning AMI, then you might want to activate the tensorflow_p36 environment as follows.
```
source activate tensorflow_p36
```
### Tornasole Path
We recommend saving tornasole outputs on S3 by passing the
flag `--tornasole_path` in the format `s3://bucket_name/prefix`.
The commands below will be shown with local path however so you can
run them immediately without having to setup S3 permissions.

### Command
```
python mnist.py --tornasole_path ~/ts_outputs/mnist
```

### Analysis
Refer [this page](../../rules/README.md) for more details on analysis.

### More
Please refer to [Tornasole Tensorflow page](../README.md).
