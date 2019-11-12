# ResNet50 Imagenet Example
We provide an example script `train_imagenet_resnet_hvd.py` which is a Tornasole-enabled TensorFlow training script for ResNet50/ImageNet.
**Please note that this script needs a GPU**.
It uses the Estimator interface of TensorFlow.
Here we show different scenarios of how to use Tornasole to
save different tensors during training for analysis.
Below are listed the changes we made to integrate these different
behaviors of Tornasole as well as example commands for you to try.

## Integrating Tornasole
Below we call out the changes for Tornasole in the above script and describe them

**Importing TornasoleTF**
```
import smdebug.tensorflow as ts
```
**Saving weights**
```
include_collections.append('weights')
```
**Saving gradients**

We need to wrap our optimizer with wrap_optimizer, and use this optimizer to minimize loss.
This will also enable us to access the gradients during analysis without having to identify which tensors out of the saved ones are the gradients.
```
opt = hook.wrap_optimizer(opt)

include_collections.append('gradients')
ts.TornasoleHook(..., include_collections=include_collections, ...)
```
**Saving relu activations by variable**
```
x = tf.nn.relu(x + shortcut)
ts.add_to_collection('relu_activations', x)
...
include_collections.append('relu_activations')
ts.TornasoleHook(..., include_collections=include_collections, ...)
```
**Saving relu activations as reductions**
```

x = tf.nn.relu(x + shortcut)
ts.add_to_collection('relu_activations', x)
...
rnc = ts.ReductionConfig(reductions=reductions, abs_reductions=abs_reductions)
...
ts.TornasoleHook(..., reduction_config=rnc, ...)
```
**Saving by regex**
```
ts.get_collection('default').include(FLAGS.tornasole_include)
include_collections.append('default')
ts.TornasoleHook(..., include_collections=include_collections, ...)
```
**Setting save interval**
```
ts.TornasoleHook(...,save_config=ts.SaveConfig(save_interval=FLAGS.tornasole_step_interval)...)
```
**Setting the right mode**

You will see in the code that the appropriate mode has been set before the train or evaluate function calls.
For example, the line:
```
hook.set_mode(ts.modes.TRAIN)
```

**Adding the hook**
```
training_hooks = []
...
training_hooks.append(hook)
classifier.train(
    input_fn=lambda: make_dataset(...),
    max_steps=nstep,
    hooks=training_hooks)
```

## Running the example
Here we provide example hyperparameters dictionaries to run this script in different scenarios from within SageMaker. You can replace the resnet_hyperparams dictionary in the notebook we provided to use the following hyperparams dictionaries to run the jobs in these scenarios.

### Run with synthetic or real data
By default the following commands run with synthetic data. If you have ImageNet data prepared in tfrecord format,
 you can pass the path to that with the parameter data_dir.

### Saving weights and gradients with Tornasole
```
hyperparams = {
    'enable_tornasole': True,
    'tornasole_save_weights': True,
    'tornasole_save_gradients': True,
    'tornasole_step_interval': 100
}
```

### Simulating gradients which 'vanish'
We simulate the scenario of gradients being really small (vanishing) by initializing weights with a small constant.

```
hyperparams = {
    'enable_tornasole': True,
    'tornasole_save_weights': True,
    'tornasole_save_gradients': True,
    'tornasole_step_interval': 100,
    'constant_initializer': 0.01
}
```
#### Rule: VanishingGradient
To monitor this condition for the first 10000 training steps, you can setup a Vanishing Gradient rule  as follows:

```
rule_specifications=[
    {
        "RuleName": "VanishingGradient",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "end-step": "10000",
        }
    }
]

```
#### Saving activations of RELU layers in full
```
hyperparams = {
    'enable_tornasole': True,
    'tornasole_save_relu_activations': True,
    'tornasole_step_interval': 200,
}
```
#### Saving activations of RELU layers as reductions
```
hyperparams = {
    'enable_tornasole': True,
    'tornasole_save_relu_activations': True,
    'tornasole_step_interval': 200,
    'tornasole_relu_reductions': 'min,max,mean,variance',
    'tornasole_relu_reductions_abs': 'mean,variance',
}
```
#### Saving weights every step
If you want to compute and track the ratio of weights and updates,
you can do that by saving weights every step as follows
```
hyperparams = {
    'enable_tornasole': True,
    'tornasole_save_weights': True,
    'tornasole_step_interval': 1
}
```
##### Rule: WeightUpdateRatio
To monitor the weights and updates during training, you can setup a WeightUpdateRatio rule as follows:

```
rule_specifications=[
    {
        "RuleName": "WeightUpdateRatio",
        "InstanceType": "ml.c5.4xlarge",
    }
]
```

##### Rule: UnchangedTensor
You can also invoke this rule to
monitor if tensors are not changing at every step. Here we are passing '.*' as the tensor_regex to monitor all tensors.
```
rule_specifications=[
    {
        "RuleName": "UnchangedTensor",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "tensor_regex": ".*"
        }
    }
]
```
