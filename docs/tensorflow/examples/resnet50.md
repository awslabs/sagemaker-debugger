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
import tornasole.tensorflow as ts
```
**Saving weights**
```
include_collections.append('weights')
```
**Saving gradients**

We need to wrap our optimizer with TornasoleOptimizer, and use this optimizer to minimize loss. 
This will also enable us to access the gradients during analysis without having to identify which tensors out of the saved ones are the gradients.
```
opt = TornasoleOptimizer(opt)

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
### Environment
Ensure you are in a python environment which has TensorFlow, TornasoleTF and TornasoleCore installed. If you followed the recommended instructions of using Amazon Deep Learning AMI, then you might want to activate the tensorflow_p36 environment as follows.
```
source activate tensorflow_p36
```
### Run with synthetic or real data
By default the following commands run with synthetic data. If you have ImageNet data prepared in tfrecord format, 
 you can pass the path to that with the flag --data_dir, such as the following:

```python train_imagenet_resnet_hvd.py --data_dir ~/data/tf-imagenet/ ...```

This flag can be appended to any of the following commands 
to make the job use real data.
### Tornasole Path
We recommend saving tornasole outputs on S3 by passing 
the flag `--tornasole_path` in the format `s3://bucket_name/prefix`. 
The commands below will be shown with local path however 
so you can run them immediately without having to setup S3 permissions.

### Example commands
#### Saving weights and gradients with Tornasole
```
python train_imagenet_resnet_hvd.py --clear_log True --enable_tornasole True \
    --tornasole_save_weights True --tornasole_save_gradients True \ 
    --tornasole_step_interval 10 \
    --tornasole_path ~/ts_outputs/default
```
#### Simulating gradients which 'vanish'
We simulate the scenario of gradients being really small (vanishing) by initializing weights with a small constant. 
```
python train_imagenet_resnet_hvd.py --clear_log True --enable_tornasole True \
    --tornasole_save_weights True --tornasole_save_gradients True \ 
    --tornasole_step_interval 10 \
    --constant_initializer 0.01 \
    --tornasole_path ~/ts_outputs/vanishing  
``` 

##### Rule: VanishingGradient
You can monitor vanishing gradients by doing the following
```
python -m tornasole.rules.rule_invoker --trial-dir ~/ts_outputs/vanishing --rule-name VanishingGradient
``` 
#### Saving activations of RELU layers in full
```
python train_imagenet_resnet_hvd.py --clear_log True  --enable_tornasole True \
    --tornasole_save_relu_activations True \
    --tornasole_step_interval 10 \
    --tornasole_path ~/ts_outputs/full_relu_activations
```
#### Saving activations of RELU layers as reductions
```
python train_imagenet_resnet_hvd.py --clear_log True  --enable_tornasole True \
    --tornasole_save_relu_activations True \
    --tornasole_relu_reductions min max mean variance \
    --tornasole_relu_reductions_abs mean variance \
    --tornasole_step_interval 10 \
    --tornasole_path ~/ts_outputs/reductions_relu_activations  
```
#### Saving weights every step
If you want to compute and track the ratio of weights and updates, 
you can do that by saving weights every step as follows 
```
python train_imagenet_resnet_hvd.py --clear_log True --enable_tornasole True \
    --tornasole_save_weights True \
    --tornasole_step_interval 1 \
    --tornasole_path ~/ts_outputs/weights
```
##### Rule: WeightUpdateRatio 
You can invoke the rule to 
monitor the ratio of weights to updates every step. 
A quick way to invoke the rule is like this: 
```
python -m tornasole.rules.rule_invoker --trial-dir ~/ts_outputs/weights --rule-name WeightUpdateRatio
```
If you want to customize the thresholds, you can pass the arguments taken by the rule as command line arguments above. 

##### Rule: UnchangedTensor
You can also invoke this rule to 
monitor if tensors are not changing at every step. Here we are passing '.*' as the tensor_regex to monitor all tensors.
```
python -m tornasole.rules.rule_invoker --trial-dir ~/ts_outputs/weights --rule-name UnchangedTensor --tensor_regex .*
```

#### Running with tornasole disabled
```
python train_imagenet_resnet_hvd.py --clear_log True 
```
### More
Please refer to [Tornasole Tensorflow page](../README.md) and the various flags in the script to customize the behavior further.
Refer [this page](../../rules/README.md) for more details on analysis. 