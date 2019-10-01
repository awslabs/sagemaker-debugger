# Tornasole for Pytorch
Tornasole is an upcoming AWS service designed to be a debugger
for machine learning models. It lets you go beyond just looking
at scalars like losses and accuracies during training and
gives you full visibility into all tensors 'flowing through the graph'
during training or inference.

Using Tornasole is a two step process:

**Saving tensors**
This needs the `tornasole` package built for the appropriate framework. This package lets you collect the tensors you want at the frequency
that you want, and save them for analysis. Sagemaker containers provided to you already have this package installed.

**Analysis**
Please refer to [this page](../../rules/DeveloperGuide_Rules.md) for more details about how to run rules and other analysis
on tensors collection from the job. The analysis of these tensors can be done on a separate machine
in parallel with the training job.

## Quickstart
Integrating Tornasole into the training job can be accomplished by following steps below.

### Import the tornasole_hook package
Import the TornasoleHook class along with other helper classes in your training script as shown below

```
from tornasole.pytorch import TornasoleHook
from tornasole.pytorch import Collection
from tornasole import SaveConfig
import tornasole.pytorch as ts
```

### Instantiate and initialize tornasole hook
Then create the TornasoleHook by specifying what you want
to save, when you want to save them and
where you want to save them. Note that for Sagemaker, you always need to specify the out_dir as `/opt/ml/output/tensors`. In the future, we will make this the default in Sagemaker environments.

```
    # Create SaveConfig that instructs engine to log graph tensors every 10 steps.
    save_config = SaveConfig(save_interval=10)
    # Create a hook that logs tensors of weights, biases and gradients while training the model.
    hook = TornasoleHook(out_dir='/opt/ml/output/tensors', save_config=save_config)
```

For additional details on TornasoleHook, SaveConfig and Collection please refer to the [API documentation](api.md)

### Register Tornasole hook to the model before starting of the training.

Here is a sample PyTorch model you may use if you wish (this is enclosed in the
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.add_module('fc1', nn.Linear(20,500))
        self.add_module('relu1', nn.ReLU())
        self.add_module('fc2', nn.Linear(500, 10))
        self.add_module('relu2', nn.ReLU())
        self.add_module('fc3', nn.Linear(10, 4))
    def forward(self, x_in):
        fc1_out = self.fc1(x_in)
        relu1_out = self.relu1(fc1_out)
        fc2_out = self.fc2(relu1_out)
        relu2_out = self.relu2(fc2_out)
        fc3_out = self.fc3(relu2_out)
        out = F.log_softmax(fc3_out, dim=1)
        return out

def create_model():
    device = torch.device("cpu")
    return Net().to(device)
```
After creating or loading the desired model, users can register the hook with the model as shown below.

```
net = create_model()
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

- `outdir`: This represents the path to which the outputs of tornasole will be written to. Note that for Sagemaker, you always need to specify the out_dir as `/opt/ml/output/tensors`. In the future, we will make this the default in Sagemaker environments.
- `save_config`: This is an object of [SaveConfig](#saveconfig). The SaveConfig allows user to specify when the tensors are to be stored. User can choose to specify the number of steps or the intervals of steps when the tensors will be stored.
- `include_collections`: This represents the [collections](#collection) to be saved. Each collection can have its own SaveConfig item.

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

Refer [API](api.md) for all methods available when using collections such as setting SaveConfig,
ReductionConfig for a specific collection, or retrieving all collections.

Please refer to [creating a collection](#creating-a-collection) to get overview of how to create collection and adding tensors to collection.

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
import tornasole.pytorch as ts
hook = ts.TornasoleHook(..., reduction_config=ts.ReductionConfig(norms=['l1']), ...)
```
Refer [API](api.md) for a full list of the reductions available.


### How to save tensors

There are different ways to save tensors when using Tornasole.
Tornasole provides easy ways to save certain standard tensors by way of default collections (a Collection represents a group of tensors).
Examples of such collections are 'weights', 'gradients'.
Besides these tensors, you can save tensors by name or regex patterns on those names.
Users can also specify a certain module in the model to save the inputs and outputs of that module.
This section will take you through these ways in more detail.

#### Default Collections
Currently, Tornasole creates Collection objects for 'weights' and 'gradients' by default for every run.
These collections store the tensors that are corresponding trainable parameters and their gradients.

#### Custom Collections
You can also create any other customized collection yourself.
You can create new collections as well as modify existing collections

##### Creating a collection
Each collection should have a unique name (which is a string). Users can create or retrieve the collection by name as follows.

```
weight_collection = ts.get_collection('weight')
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
custom_collect = ts.get_collection("ReluActivation")
custom_collect.include(["relu*", "input_*"])
```

###### Adding tensors from torch.nn Module
If users want to log the inputs and outputs of a particular module, they can do so by creating a collection as shown below. For the example below, assume conv2d is the module we wish to log the inputs and outputs of

```
module_collection = ts.get_collection('Conv2DModule')
module_collection.add_module_tensors(conv2d, inputs=True, outputs=True)
```

For creating this collection, users must have access to the module object whose inputs and outputs are to be logged.

#### Saving All Tensors
Tornasole makes it easy to save all the tensors in the model. You just need to set the flag `save_all=True` when creating the hook.
This creates a collection named 'all' and saves all the tensors under that collection.
**NOTE : Storing all the tensors will slow down the training and will increase the storage consumption.**


### More Examples

#### Logging the weights, biases, and gradients of the model

Here is how to create a hook for this purpose.

```
 # Create Tornasole hook. The initializations of hook determines which tensors
 # are logged while training is in progress.
 # Following function shows the default initilization that enables logging of
 # weights, biases and gradients in the model.
 def create_tornasole_hook(output_dir):
    # Create a SaveConfig that determines tensors from which steps are to be stored.
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3.
    save_config = SaveConfig(save_steps=[1, 2, 3])
    # Create a hook that logs ONLY weights, biases, and gradients while training the model.
    hook = TornasoleHook(out_dir=output_dir, save_config=save_config)
    return hook
```

Here is how to register the hook

```
# Assume your model is called net
hook = create_tornasole_hook(output_dir)
hook.register_hook(net)
```

#### Logging the inputs and output of a model along with weights and gradients
In order to achieve this we would need to create a collection as follows

```
# In order to log the inputs and output of a module, we will create a collection as follows:
get_collection('l_mod').add_module_tensors(module, inputs=True, outputs=True)
```

The name of the Collection is "l_mod". We have created it around the top level module of the model which represents the whole complete model itself to this collection. As a result this collection will contain tensors that were inputs and outputs of this module (e.g. the model itself) at corresponding training steps.
The following code shows how to initialize the hook with the above collection.

```
def create_tornasole_hook(output_dir, module):
    # The names of input and output tensors of a module are in following format
    # Inputs :  <module_name>_input_<input_index>, and
    # Output :  <module_name>_output
    # In order to log the inputs and output of a module, we will create a collection as follows:
    assert module is not None
    get_collection('l_mod').add_module_tensors(module, inputs=True, outputs=True)

    # Create a hook that logs weights, biases, gradients and inputs outputs of model while training.
    hook = TornasoleHook(out_dir=output_dir, save_config=SaveConfig(save_steps=[i * 10 for i in range(5)]),
			    include_collections=['weights', 'gradients', 'biases','l_mod'])
```

Here is how to register the above hook.

```
# Assume your model is called net
hook = create_tornasole_hook(output_dir=output_dir, module=net)
hook.register_hook(net)
```

#### Logging the inputs and output of a module in the model along with weights and gradients
Follow the same procedure as above; just pass
in the appropriate module into `create_tornasole_hook`.

#### Saving all tensors in the model
For saving all the tensors users not required to create a special collection.
Users can set the _save_all_ flag while creating a TornasoleHook object in the manner shown below.

```
 # Create Tornasole hook. The initializations of hook determines which tensors
 # are logged while training is in progress.
 # Following function shows the default initilization that enables logging of
 # weights, biases and gradients in the model.
 def create_tornasole_hook(output_dir):
    # Create a SaveConfig that determines tensors from which steps are to be stored.
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3.
    save_config = SaveConfig(save_steps=[1, 2, 3])
    # Create a hook that logs weights, biases, gradients, module inputs, and module outputs of all layers while training the model.
    hook = TornasoleHook(out_dir=output_dir, save_config=save_config, saveall=True)
    return hook
```

Here is how to register the hook

```
# Assume your model is called net
hook = create_tornasole_hook(output_dir)
hook.register_hook(net)
```
All tensors will be saved as part of a collection named 'all'.

## Analyzing the Results

This library enables users to collect the desired tensors at desired frequency while the PyTorch job is running.
The tensor data generated during this job can be analyzed with various rules
that check for vanishing gradients, exploding gradients, etc.
For details regarding how to analyze the tensor data, usage of existing rules or
writing new rules, please refer to [Rules documentation](../../rules/DeveloperGuide_Rules.md).


## FAQ
#### Logging
You can control the logging from Tornasole by setting the appropriate
level for the python logger `tornasole` using either of the following approaches.

**In Python code**
```
import logging
logging.getLogger('tornasole').setLevel = logging.INFO
```

**Using environment variable**
You can also set the environment variable `TORNASOLE_LOG_LEVEL` as below

```
export TORNASOLE_LOG_LEVEL=INFO
```
Log levels available are 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL', 'OFF'.

## ContactUs
We would like to hear from you. If you have any question or feedback, please reach out to us tornasole-users@amazon.com

## License
This library is licensed under the Apache 2.0 License.
