# Tornasole for MXNet
Tornasole is an upcoming AWS service designed to be a debugger
for machine learning models. It lets you go beyond just looking
at scalars like losses and accuracies during training and
gives you full visibility into all tensors 'flowing through the graph'
during training or inference.

Using Tornasole is a two step process:

**Saving tensors**
This needs the `tornasole` package built for the appropriate framework. This package lets you collect the tensors you want at the frequency
that you want, and save them for analysis.
Please follow the appropriate Readme page to install the correct version. This page is for using Tornasole with MXNet.

**Analysis**
Please refer to [this page](../rules/README.md) for more details about how to run rules and other analysis
on tensors collection from the job. That said, we do provide a few example analysis commands below
so as to provide an end to end flow. The analysis of these tensors can be done on a separate machine
in parallel with the training job.

## Installation
#### Prerequisites
- **Python 3.6**
- Tornasole can work in local mode or remote(s3) mode. You can skip this, if you want to try [local mode example](#tornasole-local-mode-example).
This is necessary to setup if you want to try [s3 mode example](#tornasole-s3-mode-example).
For running in S3 mode, you need to make sure that instance you are using has proper credentials set to have S3 write access.
Try the below command -
```
 aws s3 ls
```
If you see errors, then most probably your credentials are not properly set.
Please follow [FAQ on S3](#s3access) to make sure that your instance has proper S3 access.

- We recommend using the `mxnet_p36` conda environment on EC2 machines launched with the AWS Deep Learning AMI.
You can activate this by doing: `source activate mxnet_p36`.

- If you are not using the above environment, please ensure that you have the MXNet framework installed.

#### Instructions
**Make sure that your aws account is whitelisted for Tornasole. [ContactUs](#contactus)**.

Once your account is whitelisted, you should be able to install the `tornasole` package built for MXNet as follows:

```
aws s3 sync s3://tornasole-external-preview-use1/sdk/ts-binaries/tornasole_mxnet/py3/latest/ tornasole_mxnet/
pip install tornasole_mxnet/*
```

**Please note** : If, while installing tornasole, you get a version conflict issue between botocore and boto3,
you might need to run the following
```
pip uninstall -y botocore boto3 aioboto3 aiobotocore && pip install botocore==1.12.91 boto3==1.9.91 aiobotocore==0.10.2 aioboto3==6.4.1
```

## Quickstart
If you want to quickly run some examples, you can jump to [examples](#examples) section. You can also see this [mnist notebook example](../../examples/mxnet/notebooks/mnist/SimpleInteractiveAnalysis.ipynb) to see tornasole working.

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

## Examples
#### Simple CPU training

##### Tornasole local mode example
The example [mnist\_gluon\_vg\_demo.py](../../examples/mxnet/scripts/mnist_gluon_vg_demo.py) is implemented to show how Tornasole is useful in detecting the vanishing gradient problem. The learning_rate and momentum in this example are set in a such way that the training will encounter the vanishing gradient issue.

```
python examples/mxnet/scripts/mnist_gluon_vg_demo.py --output-uri ~/tornasole-testing/vg-demo/trial-one
```

You can monitor the job for vanishing gradients by doing the following:

```
python -m tornasole.rules.rule_invoker --trial-dir ~/tornasole-testing/vg-demo/trial-one --rule-name VanishingGradient
```

Note: You can also try some further analysis on tensors saved by following [programming model](../rules/README.md#the-programming-model) section of our Rules README.

##### Tornasole S3 mode example

```
python examples/mxnet/scripts/mnist_gluon_vg_demo.py --output-uri s3://tornasole-testing/vg-demo/trial-one
```

You can monitor the job for vanishing gradients by doing the following:

```
python -m tornasole.rules.rule_invoker --trial-dir s3://tornasole-testing/vg-demo/trial-one --rule-name VanishingGradient
```
Note: You can also try some further analysis on tensors saved by following [programming model](../rules/README.md#the-programming-model) section of our Rules README.

## API
Please refer to [this document](api.md) for description of all the functions and parameters that our APIs support

####  Hook
TornasoleHook is the entry point for Tornasole into your program.
Some key parameters to consider when creating the TornasoleHook are the following:

- `out_dir`: This represents the path to which the outputs of tornasole will be written to under a directory with the name `out_dir`. This can be a local path or an S3 prefix of the form `s3://bucket_name/prefix`.
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
dictionary from mode to SaveConfig object.
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

- `save_interval`: This allows you to save tensors every `n` steps
- `save_steps`: Allows you to pass a list of step numbers at which tensors should be saved; overrides `start_step` and `end_step`
- `start_step`: The step at which to start saving
- `end_step`: The step at which to stop saving
- `when_nan`: A list of tensor regexes to save if they become NaN

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


### More Examples
| Example Type   | Logging Weights and Gradients   | Logging inputs and outputs of the model  | Logging inputs and outputs of a block in the model.  | Saving all tensors.   | Vanishing Gradient demo   |
| --------------- | -----------------------------  | -----------------------------  | -----------------------------  | -----------------------------  | -----------------------------  |
| Link to Example   | [mnist\_gluon\_basic\_hook\_demo.py](../../examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py)   | [mnist\_gluon\_model\_input\_output\_demo.py](../../examples/mxnet/scripts/mnist_gluon_model_input_output_demo.py)   |  [mnist\_gluon\_block\_input\_output\_demo.py](../../examples/mxnet/scripts/mnist_gluon_block_input_output_demo.py)   | [mnist\_gluon\_save\_all\_demo.py](../../examples/mxnet/scripts/mnist_gluon_save_all_demo.py)   | [mnist\_gluon\_vg\_demo.py](../../examples/mxnet/scripts/mnist_gluon_vg_demo.py)   |

#### Logging the weights and gradients of the model

The [mnist\_gluon\_basic\_hook\_demo.py](../../examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py) shows end to end example of how to create and register Tornasole hook that can log tensors of model weights and their gradients.

Here is how to create a hook for this purpose.

```
 # Create a tornasole hook. The initialization of hook determines which tensors
 # are logged while training is in progress.
 # Following function shows the default initialization that enables logging of
 # weights, biases and gradients in the model.
def create_tornasole_hook(output_s3_uri):
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3
    # (indexing starts with 0).
    save_config = SaveConfig(save_steps=[1, 2, 3])
    # Create a hook that logs weights, biases and gradients while training the model.
    hook = TornasoleHook(out_dir=output_s3_uri, save_config=save_config)
    return hook
```

Here is how to register the hook

```
 # Create a model using gluon API. The tornasole hook is currently
 # supports MXNet gluon models only.
def create_gluon_model():
    # Create Model in Gluon
    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
...
            nn.Dense(10))
    net.initialize(init=init.Xavier(), ctx=mx.cpu())
    return net
...
    # Create a Gluon Model.
    net = create_gluon_model()
...
    # Create a Gluon Model.
	net = create_gluon_model()
	hook = create_tornasole_hook(output_s3_uri)
	hook.register(net)
```

The example can be invoked as shown below. **Ensure that the s3 bucket specified in command line is accessible for read and write operations**

```
python examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py --output-uri s3://tornasole-testing/basic-mxnet-hook --trial-id trial-one
```

For detail command line help run

```
python examples/mxnet/scripts/mnist_gluon_basic_hook_demo.py --help
```

#### Logging the inputs and output of a model along with weights and gradients
The [mnist\_gluon\_model\_input\_output\_demo.py](../../examples/mxnet/scripts/mnist_gluon_model_input_output_demo.py) shows how to create and register the tornasole hook that can log the inputs and output of the model in addition to weights and gradients tensors.
In order to achieve this we would need to create a collection as follows

```
 # The names of input and output tensors of a block are in following format
 # Inputs :  <block_name>_input_<input_index>, and
 # Output :  <block_name>_output
 # In order to log the inputs and output of a model, we will create a collection as follows:
     tm.get_collection('TopBlock').add_block_tensors(block, inputs=True, outputs=True)
```

The name of the Collection is "TopBlock". We have created it around top level block of the model which represents the whole complete model itself to this collection. As a result this collection will contain tensors that were inputs and outputs of this block (e.g. model itself) at corresponding training steps.
Following code shows how to initialize the hook with the above collection.

```
 # Create a tornasole hook. The initialization of hook determines which tensors
 # are logged while training is in progress.
 # Following function shows the hook initialization that enables logging of
 # weights, biases and gradients in the model along with the inputs and outputs of the model.
def create_tornasole_hook(output_s3_uri, block):
    # Create a SaveConfig that determines tensors from which steps are to be stored.
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3.
    save_config = SaveConfig(save_steps=[1, 2, 3])
    # The names of input and output tensors of a block are in following format
    # Inputs :  <block_name>_input_<input_index>, and
    # Output :  <block_name>_output
    # In order to log the inputs and output of a model, we will create a collection as follows:
    tm.get_collection('TopBlock').add_block_tensors(block, inputs=True, outputs=True)
    # Create a hook that logs weights, biases, gradients and inputs outputs of model while training.
    hook = TornasoleHook(out_dir=output_s3_uri, save_config=save_config, include_collections=['weights', 'gradients', 'bias','TopBlock'])
    return hook
```

Here is how to register the above hook.

```
 # Create a model using gluon API. The tornasole hook is currently
 # supports MXNet gluon models only.
def create_gluon_model():
    # Create Model in Gluon
    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
...
            nn.Dense(10))
    net.initialize(init=init.Xavier(), ctx=mx.cpu())
    return net
...
net = create_gluon_model()
hook = create_tornasole_hook(output_s3_uri)
hook.register(net)
```

The example can be invoked as shown below. **Ensure that the s3 bucket specified in command line is accessible for read and write operations**

```
python examples/mxnet/scripts/mnist_gluon_model_input_output_demo.py --output-s3-uri s3://tornasole-testing/model-io-hook/trial-one
```

For detail command line help run

```
python examples/mxnet/scripts/mnist_gluon_model_input_output_demo.py --help
```

#### Logging the inputs and output of a block in the model along with weights and gradients
The [mnist\_gluon\_block\_input\_output\_demo.py](../../examples/mxnet/scripts/mnist_gluon_block_input_output_demo.py) shows how to create and register the tornasole hook that can log the inputs and output of a particular block in the model in addition to weights and gradients tensors.

**NOTE: For this type of logging the Gluon Model should not be hybridized.**
In order to achieve this we need to have access to the block object whose tensors we want to log. The following code snippet shows how we can cache the block objects while creating a model. Ensure that the model is not hybridized.

```
 # Create a model using gluon API. The tornasole hook is currently
 # supports MXNet gluon models only.
def create_gluon_model():
    # Create Model in Gluon
    child_blocks = []
    net = nn.HybridSequential()
    conv2d_0 = nn.Conv2D(channels=6, kernel_size=5, activation='relu')
    child_blocks.append(conv2d_0)
    maxpool2d_0 = nn.MaxPool2D(pool_size=2, strides=2)
...
    dense_2 = nn.Dense(10)
    child_blocks.append(dense_2)
    net.add(conv2d_0, maxpool2d_0, conv2d_1, maxpool2d_1, flatten_0, dense_0, dense_1, dense_2)
    net.initialize(init=init.Xavier(), ctx=mx.cpu())
    return net, child_blocks
```

We can then create a collection to log the input output tensors of one of the child_blocks. For example, child_block[0] is passed to *create_tornasole_hook* function as 'block' and the function creates collection for that block as shown below

```
 # Create a tornasole hook. The initialization of hook determines which tensors
 # are logged while training is in progress.
 # Following function shows the hook initialization that enables logging of
 # weights, biases and gradients in the model along with the inputs and output of the given
 # child block.
def create_tornasole_hook(output_s3_uri, block):
    # Create a SaveConfig that determines tensors from which steps are to be stored.
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3.
    save_config = SaveConfig(save_steps=[1, 2, 3])
    # The names of input and output tensors of a block are in following format
    # Inputs :  <block_name>_input_<input_index>, and
    # Output :  <block_name>_output
    # In order to log the inputs and output of a model, we will create a collection as follows
    tm.get_collection(block.name).add_block_tensors(block, inputs=True, outputs=True)
    # Create a hook that logs weights, biases, gradients and inputs outputs of model while training.
    hook = TornasoleHook(out_dir=output_s3_uri, save_config=save_config, include_collections=[
        'weights', 'gradients', 'bias', block.name])
    return hook
```

The name of the Collection is kept same as the name of block.
We have created it around a block in the model.
As a result this collection will contain tensors that were inputs and outputs of
this block (e.g. Conv2D block) at corresponding training steps.


Here is how to register the above hook.

```
    # Create a Gluon Model.
    net,child_blocks = create_gluon_model()
    # Create a tornasole hook for logging the desired tensors.
    # The output_s3_uri is a the URI for the s3 bucket where the tensors will be saved.
    # The trial_id is used to store the tensors from different trials separately.
    output_s3_uri=opt.output_s3_uri
    # For creating a tornasole hook that can log inputs and output of the specific child block in the model,
    # we will pass the desired block object to the create_tornasole_hook function.
    # In the following case, we are attempting log inputs and output of the first Conv2D block.
    hook = create_tornasole_hook(output_s3_uri, child_blocks[0])
    # Register the hook to the top block.
    hook.register_hook(net)
```

The example can be invoked as shown below.
**Ensure that the s3 bucket specified in command line is accessible for read and write operations**

```
python examples/mxnet/scripts/mnist_gluon_block_input_output_demo.py --output-s3-uri s3://tornasole-testing/block-io-mxnet-hook/trial-one
```

For detail command line help run

```
python examples/mxnet/scripts/mnist_gluon_block_input_output_demo.py --help
```

#### Saving all tensors in the model
The [mnist\_gluon\_save_all\_demo.py](../../examples/mxnet/scripts/mnist_gluon_save_all_demo.py) shows how to store every tensor in the model.
As mentioned above, for saving all the tensors users not required to create a special collection. Users can set _save_all_ flag while creating TornasoleHook as shown below.

```
 # Create a tornasole hook. The initialization of hook determines which tensors
 # are logged while training is in progress.
 # Following function shows the initialization of tornasole hook that enables logging of
 # all the tensors in the model.
def create_tornasole_hook(output_s3_uri):
    # Create a SaveConfig that determines tensors from which steps are to be stored.
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3.
    save_config = SaveConfig(save_steps=[1, 2, 3])
    # Create a hook that logs all the tensors seen while training the model.
    hook = TornasoleHook(out_dir=output_s3_uri, save_config=save_config, save_all=True)
    return hook
```

Here is how to register the above hook.

```
 # Create a model using gluon API. The tornasole hook is currently
 # supports MXNet gluon models only.
def create_gluon_model():
    # Create Model in Gluon
    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
...
            nn.Dense(10))
    net.initialize(init=init.Xavier(), ctx=mx.cpu())
    return net
...
net = create_gluon_model()
hook = create_tornasole_hook(output_s3_uri)
hook.register(net)
```

The example can be invoked as shown below. **Ensure that the s3 bucket specified in command line is accessible for read and write operations**

```
python examples/mxnet/scripts/mnist_gluon_save_all_demo.py --output-s3-uri s3://tornasole-testing/saveall-mxnet-hook/trial-one
```
For detail command line help run

```
python examples/mxnet/scripts/mnist_gluon_save_all_demo.py --help
```

#### Example demonstrating the vanishing gradient
The example [mnist\_gluon\_vg\_demo.py](../../examples/mxnet/scripts/mnist_gluon_vg_demo.py) is implemented to show how Tornasole is useful in detecting the vanishing gradient problem. The learning_rate and momentum in this example are set in a such way that the training will encounter the vanishing gradient issue.
The example can be invoked as follows

```
python examples/mxnet/scripts/mnist_gluon_vg_demo.py --output-uri s3://tornasole-testing/vg-demo/trial-one
```

## Analyzing the Results

This library enables users to collect the desired tensors at desired frequency while MXNet job is running.
The tensor data generated during this job can be analyzed with various
rules that check for vanishing gradients, exploding gradients, etc.
For example, the [mnist\_gluon\_vg\_demo.py](../../examples/mxnet/scripts/mnist_gluon_vg_demo.py)
has the vanishing gradient issue. When the tensors generated by this example are
analyzed by 'VanishingGradient' rule, it shows in which steps the model encounters the vanishing gradient issue.

```
python -m tornasole.rules.rule_invoker --trial-dir s3://tornasole-testing/vg-demo/trial-one --rule-name VanishingGradient
```

For details regarding how to analyze the tensor data, usage of existing rules or writing new rules,
please refer to [Rules documentation](../rules/README.md).


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

#### S3Access
The instance running tornasole in s3 mode needs to have s3 access. There are different ways to provide an instance to your s3 account.
- If you using EC2 instance, you should launch your instance with proper iam role to access s3. https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html
- If you are using mac or other machine, you can create a IAM user for your account to have s3 access by following this guide (https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html) and then configure your instance to use your AWS_ACCESS_KEY_ID AND AWS_SECRET_KEY_ID by using doc here https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
- Once you are done configuring, please verify that below is working and buckets returned are from the account and region you want to use.
```
aws s3 ls
```

## ContactUs
We would like to hear from you. If you have any question or feedback, please reach out to us tornasole-users@amazon.com

## License
This library is licensed under the Apache 2.0 License.
