# Tornasole for MXNet
Tornasole is designed to be a debugger for machine learning models. It lets you go beyond just looking
at scalars like losses and accuracies during training and
gives you full visibility into all tensors 'flowing through the graph'
during training or inference.


## Quickstart
If you want to quickly run an end to end example, please refer to [mnist notebook example](examples/notebooks/mxnet.ipynb) to see tornasole working.

Integrating Tornasole into the training job can be accomplished by following steps below.

### Import the tornasole_hook package
Import the TornasoleHook class along with other helper classes in your training script as shown below

```
from tornasole.mxnet.hook import TornasoleHook
from tornasole.mxnet import SaveConfig, Collection
```

### Instantiate and initialize tornasole hook

```
    # Create SaveConfig that instructs engine to log graph tensors every 10 steps.
    save_config = SaveConfig(save_interval=10)
    # Create a hook that logs tensors of weights, biases and gradients while training the model.
    output_s3_uri = 's3://my_mxnet_training_debug_bucket/12345678-abcd-1234-abcd-1234567890ab'
    hook = TornasoleHook(out_dir=output_s3_uri, save_config=save_config)
```

Using the _Collection_ object and/or _include\_regex_ parameter of TornasoleHook , users can control which tensors will be stored by the TornasoleHook.
The section [How to save tensors](#how-to-save-tensors) explains various ways users can create _Collection_ object to store the required tensors.

The _SaveConfig_ object controls when these tensors are stored. The tensors can be stored for specific steps or after certain interval of steps. If the save\_config parameter is not specified, the TornasoleHook will store tensors after every 100 steps.

For additional details on TornasoleHook, SaveConfig and Collection please refer to the [API documentation](api.md)

### Register Tornasole hook to the model before starting of the training.

#### NOTE: The tornasole hook can only be registered to Gluon Non-hybrid models.

After creating or loading the desired model, users can register the hook with the model as shown below.

```
net = create_gluon_model()
 # Apply hook to the model (e.g. instruct engine to recognize hook configuration
 # and enable mode in which engine will log graph tensors
hook.register_hook(net)
```

#### Set the mode
Set the mode you are running the job in. This helps you group steps by mode,
for easier analysis.
If you do not specify this, it saves steps under a `default` mode.
```
hook.set_mode(ts.modes.TRAIN)
```

## API
Please refer to [this document](api.md) for description of all the functions and parameters that our APIs support

####  Hook
TornasoleHook is the entry point for Tornasole into your program.
Some key parameters to consider when creating the TornasoleHook are the following:

- `out_dir`: This represents the path to which the outputs of tornasole will be written to. Note that for Sagemaker, you always need to specify the out_dir as `/opt/ml/output/tensors`. In the future, we will make this the default in Sagemaker environments.
- `save_config`: This is an object of [SaveConfig](#saveconfig). The SaveConfig allows user to specify when the tensors are to be stored. User can choose to specify the number of steps or the intervals of steps when the tensors will be stored. If not specified, it defaults to a SaveConfig which saves every 100 steps.
- `include_collections`: This represents the [collections](#collection) to be saved. With this parameter, user can control which tensors are to be saved.
- `include_regex`: This represents the regex patterns of names of tensors to save. With this parameter, user can control which tensors are to be saved.

**Examples**

- Save weights and gradients every 100 steps to an S3 location

```
import tornasole.mxnet as tm
tm.TornasoleHook(out_dir='s3://tornasole-testing/trial_job_dir',
                 save_config=tm.SaveConfig(save_interval=100),
                 include_collections=['weights', 'gradients'])
```

- Save custom tensors by regex pattern to a local path

```
import tornasole.mxnet as tm
tm.TornasoleHook(out_dir='/home/ubuntu/tornasole-testing/trial_job_dir',
                 include_regex=['relu*'])
```

Refer [API](api.md) for all parameters available and detailed descriptions.

### Mode
A machine learning job can be executing steps in multiple modes, such as training, evaluating, or predicting.
Tornasole provides you the construct of a `mode` to keep data from these modes separate
and make it easy for analysis. To leverage this functionality you have to
call the `set_mode` function of hook such as the following call `hook.set_mode(modes.TRAIN)`.
The different modes available are `modes.TRAIN`, `modes.EVAL` and `modes.PREDICT`.


If the mode was not set, all steps will be available together.

You can choose to have different save configurations (SaveConfigMode)
for different modes. You can configure this by passing a
dictionary from mode to SaveConfigMode object.
The hook's `save_config` parameter accepts such a dictionary, as well as collection's `set_save_config` method.
```
from tornasole.tensorflow import TornasoleHook, get_collection, modes, SaveConfigMode
scm = {modes.TRAIN: SaveConfigMode(save_interval=100),
        modes.EVAL: SaveConfigMode(save_interval=10)}

hook = TornasoleHook(...,
                     save_config=scm,
                     ...)
```

```
from tornasole.tensorflow import get_collection, modes, SaveConfigMode
get_collection('weights').set_save_config({modes.TRAIN: SaveConfigMode(save_interval=10),
                                           modes.EVAL: SaveConfigMode(save_interval=1000)}
```
#### Collection
Collection object helps group tensors for easier handling of tensors being saved.
A collection has its own list of tensors, include regex patterns, [reduction config](#reductionconfig) and [save config](#saveconfig).
This allows setting of different save and reduction configs for different tensors.
These collections are then also available during analysis.
Tornasole will save the value of tensors in collection, if the collection is included in `include_collections` param of the [hook](#hook).

Refer [API](api.md) for all methods available when using collections such
as setting SaveConfig,
ReductionConfig for a specific collection, or retrieving all collections.

Please refer to [creating a collection](#creating-a-collection) to get overview of how to
create collection and adding tensors to collection.

#### SaveConfig
SaveConfig class allows you to customize the frequency of saving tensors.
The hook takes a SaveConfig object which is applied as
default to all tensors included.
A collection can also have its own SaveConfig object which is applied
to the tensors belonging to that collection.

SaveConfig also allows you to save tensors when certain tensors become nan.
This list of tensors to watch for is taken as a list of strings representing names of tensors.

The parameters taken by SaveConfig are:

- `save_interval`: This allows you to save tensors every `n` steps; when `step_num % save_interval == 0`.
- `start_step`: The step at which to start saving (inclusive), defaults to 0.
- `end_step`: The step at which to stop saving (exclusive), default to None/Infinity.
- `save_steps`: Allows you to pass a list of step numbers at which tensors should be saved; overrides `save_interval`, `start_step`, and `end_step`.
- `when_nan`: List of tensor regexes; will save tensors whenever any of these tensors becomes NaN or infinite.
If this is passed along with either `save_steps` or `save_interval`, then tensors will be saved whenever this list of tensors is not finite
as well as when a particular step should be saved based on the above two parameters.

Refer [API](api.md) for all parameters available and detailed descriptions for them, as well as example SaveConfig objects.

#### ReductionConfig
ReductionConfig allows the saving of certain reductions of tensors instead
of saving the full tensor. By reduction here we mean an operation that converts the tensor to a scalar. The motivation here is to reduce the amount of data
saved, and increase the speed in cases where you don't need the full tensor.
The reduction operations which are computed in the training process and then saved.
During analysis, these are available as reductions of the original tensor.
**Please note that using reduction config means that you will not have
the full tensor available during analysis, so this can restrict what you can do with the tensor saved.**
The hook takes a ReductionConfig object which is applied as default to all tensors included.
A collection can also have its own ReductionConfig object which is applied
to the tensors belonging to that collection.

**Examples**

- ```ReductionConfig(abs_reductions=['min','max','mean'])``` Save min, max, mean on absolute values of the tensors included

- ```ReductionConfig(reductions=['min','max','mean'])``` Save min, max, mean of the tensors included

- ```ReductionConfig(norms=['l1'])``` Saves l1 norm of the tensors included


These reduction config instances can be passed to the hook as follows

```
	import tornasole.mxnet as tm
	global_reduce_config = tm.ReductionConfig(reductions=["max", "mean"])
	hook = tm.TornasoleHook(out_dir=out_dir, save_config=global_save_config,reduction_config=global_reduce_config)
```

Or ReductionConfig can be specified for an individual collection as follows

```
import tornasole.mxnet as tm
tm.get_collection("ReluActivation").include(["relu*"])
tm.get_collection("ReluActivation").set_save_config(SaveConfig(save_steps=[4,5,6]))
tm.get_collection("ReluActivation").set_reduction_config(ReductionConfig(reductions=["min"], abs_reductions=["max"]))
...
tm.get_collection("flatten").include(["flatten*"])
tm.get_collection("flatten").set_save_config(SaveConfig(save_steps=[4,5,6]))
tm.get_collection("flatten").set_reduction_config(ReductionConfig(norms=["l1"], abs_norms=["l2"]))
hook = TornasoleHook(out_dir=out_dir, include_collections=['weights', 'bias','gradients',
                                                    'default', 'ReluActivation', 'flatten'])
```

Refer [API](api.md) for a list of the reductions available as well as examples.


### How to save tensors

There are different ways to save tensors when using Tornasole.
Tornasole provides easy ways to save certain standard tensors by way of default collections (a Collection represents a group of tensors).
Examples of such collections are 'weights', 'gradients', 'bias' and 'default'.
Besides the tensors in above default collections, you can save tensors by name or regex patterns on those names.
Users can also specify a certain block in the model to save the inputs and outputs of that block.
This section will take you through these ways in more detail.

#### Saving the tensors with _include\_regex_
The TornasoleHook API supports _include\_regex_ parameter. The users can specify a regex pattern with this pattern. The TornasoleHook will store the tensors that match with the specified regex pattern. With this approach, users can store the tensors without explicitly creating a Collection object. The specified regex pattern will be associated with 'default' Collection and the SaveConfig object that is associated with the 'default' collection.

#### Default Collections
Currently, the tornasole\_mxnet hook creates Collection objects for 'weights', 'gradients', 'bias' and 'default'. These collections contain the regex pattern that match with tensors of type weights, gradient and bias. The regex pattern for the 'default' collection is set when user specifies _include\_regex_ with TornasoleHook or sets the _SaveAll=True_.  These collections use the SaveConfig parameter provided with the TornasoleHook initialization. The TornasoleHook will store the related tensors, if user does not specify any special collection with _include\_collections_ parameter. If user specifies a collection with _include\_collections_ the above default collections will not be in effect.

#### Custom Collections
You can also create any other customized collection yourself.
You can create new collections as well as modify existing collections

##### Creating a collection
Each collection should have a unique name (which is a string). You can create collections by invoking helper methods as described in the [API](api.md) documentation

```
import tornasole.mxnet as tm
tm.get_collection('weights').include(['weight'])
```

##### Adding tensors
Tensors can be added to a collection by either passing an include regex parameter to the collection.
If you don't know the name of the tensors you want to add, you can also add the tensors to the collection
by the variables representing the tensors in code. The following sections describe these two scenarios.

###### Adding tensors by regex
If you know the name of the tensors you want to save and can write regex
patterns to match those tensornames, you can pass the regex patterns to the collection.
The tensors which match these patterns are included and added to the collection.

```
import tornasole.mxnet as tm
tm.get_collection('ReluActivation').include(["relu*", "input_*"])
```

###### Adding tensors from Gluon block
If users want to log the inputs and outputs of a particular block in the Gluon model. They can do so by creating a collection as shown below.

```
import tornasole.mxnet as tm
tm.get_collection('Conv2DBlock').add_block_tensors(conv2d, inputs=True, outputs=True)
```

For creating this collection, users must have access to the block object whose inputs and outputs are to be logged.

#### Saving All Tensors
Tornasole makes it easy to save all the tensors in the model. You just need to set the flag `save_all=True` when creating the hook. This creates a collection named 'all' and saves all the tensors under that collection.
**NOTE : Storing all the tensors will slow down the training and will increase the storage consumption.**


## ContactUs
We would like to hear from you. If you have any question or feedback, please reach out to us tornasole-users@amazon.com

## License
This library is licensed under the Apache 2.0 License.
