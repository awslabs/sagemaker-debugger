# Tornasole for TensorFlow
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
If you want to quickly run an end to end example in Sagemaker,
you can jump to the notebook [examples/notebooks/tensorflow.ipynb](examples/notebooks/tensorflow.ipynb).

Integrating Tornasole into your job is as easy as adding the following lines of code:

### Session based training
We need to add Tornasole Hook and use it to create a monitored session for the job.
First, we need to import `smdebug.tensorflow`.
```
import smdebug.tensorflow as ts
```
Then create the TornasoleHook by specifying what you want
to save, when you want to save them and
where you want to save them. Note that for Sagemaker,
you always need to specify the out_dir as `/opt/ml/output/tensors`. In the future,
we will make this the default in Sagemaker environments.
```
hook = ts.TornasoleHook(out_dir='/opt/ml/output/tensors',
                        include_collections=['weights','gradients'],
                        save_config=ts.SaveConfig(save_interval=2))
```

Set the mode you are running the job in. This helps you group steps by mode,
for easier analysis.
If you do not specify this, it saves steps under a `GLOBAL` mode.
```
hook.set_mode(ts.modes.TRAIN)
```

Wrap your optimizer with wrap_optimizer so that
Tornasole can identify your gradients and automatically
provide these tensors as part of the `gradients` collection.
Use this new optimizer to minimize the loss.
```
optimizer = hook.wrap_optimizer(optimizer)
```

Create a monitored session with the above hook, and use this for executing your TensorFlow job.
```
sess = tf.train.MonitoredSession(hooks=[hook])
```

### Estimator based training
We need to create TornasoleHook and provide it to the estimator's train, predict or evaluate methods.
First, we need to import `smdebug.tensorflow`.
```
import smdebug.tensorflow as ts
```
Then create the TornasoleHook by specifying what you want
to save, when you want to save them and
where you want to save them. Note that for Sagemaker,
you always need to specify the out_dir as `/opt/ml/output/tensors`. In the future,
we will make this the default in Sagemaker environments.
```
hook = ts.TornasoleHook(out_dir='/opt/ml/output/tensors',
                        include_collections = ['weights','gradients'],
                        save_config = ts.SaveConfig(save_interval=2))
```
Set the mode you are running the job in. This helps you group steps by mode, for easier
analysis.
If you do not specify this, it saves steps under a `GLOBAL` mode.
```
hook.set_mode(ts.modes.TRAIN)
```
Wrap your optimizer with wrap_optimizer so that
Tornasole can identify your gradients and automatically
provide these tensors as part of the `gradients` collection.
Use this new optimizer to minimize the loss.
```
opt = hook.wrap_optimizer(opt)
```
Now pass this hook to the estimator object's train, predict or evaluate methods, whichever ones you want to monitor.
```
classifier = tf.estimator.Estimator(...)

classifier.train(input_fn, hooks=[hook])
classifier.predict(input_fn, hooks=[hook])
classifier.evaluate(input_fn, hooks=[hook])
```
Refer [TF Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) for information on the train, predict, evaluate functions.

#### Note
**Keras** support is Work in Progress. Please stay tuned!
We will also support **Eager** mode in the future.


## Tornasole TensorFlow Concepts
In this section we briefly introduce the main constructs of the Tornasole TF API and some parameters important for their construction.
Please refer to [this document](api.md) for description of all the functions and parameters that our APIs support.

####  Hook
TornasoleHook is the entry point for Tornasole into your program.
It's a subclass of `tf.train.SessionRunHook` and can be used where that is suitable,
such as MonitoredSession and Estimator's train/predict/evaluate methods.
Some key parameters to consider when creating the TornasoleHook are the following:
- `out_dir`: This represents the path to which the outputs of tornasole will be written to. Note that for Sagemaker, you always need to specify the out_dir as `/opt/ml/output/tensors`. In the future, we will make this the default in Sagemaker environments.
- `save_config`: The hook takes a SaveConfig object which controls when tensors are saved.
It defaults to a SaveConfig which saves every 100 steps.
- `include_regex`: This represents the regex patterns of names of tensors to save
- `include_collections`: This represents the collections to be saved


It also has an important method which can be used to set the appropriate mode.
Modes can refer to 'training', 'evaluation' or 'prediction'. They can be set as follows:
```hook.set_mode(ts.modes.TRAIN)```, ```hook.set_mode(ts.modes.EVAL)``` or ```hook.set_mode(ts.modes.PREDICT)```.
This allows you to group steps by mode which allows for clearer analysis. Tornasole
also allows you to see a global ordering of steps which makes it clear after how many training
steps did a particular evaluation step happen. If you do not set this mode, all steps are saved under
a `default` mode.

**Examples**
- Save weights and gradients every 100 steps to an S3 location
```
import smdebug.tensorflow as ts
ts.TornasoleHook(out_dir='/opt/ml/output/tensors',
                 save_config=ts.SaveConfig(save_interval=100),
                 include_collections=['weights', 'gradients'])
```

- Save custom tensors by regex pattern to a local path
```
import smdebug.tensorflow as ts
ts.TornasoleHook(out_dir='/opt/ml/output/tensors',
                 include_regex=['loss*'])
```
Refer [API](api.md) for all parameters available and their detailed descriptions.

#### Mode
A machine learning job can be executing steps in multiple modes, such as training, evaluating, or predicting.
Tornasole provides you the construct of a `mode` to keep data from these modes separate
and make it easy for analysis. To leverage this functionality you have to
call the `set_mode` function of hook such as the following call `hook.set_mode(modes.TRAIN)`.
The different modes available are `modes.TRAIN`, `modes.EVAL` and `modes.PREDICT`.

If the mode was not set, all steps will be available together.

You can choose to have different save configurations (SaveConfigMode)
for different modes. You can configure this by passing a
dictionary from mode to SaveConfigMode object.
The hook's `save_config` parameter accepts such a dictionary, as well as collection's `save_config` property.
```
from smdebug.tensorflow import TornasoleHook, get_collection, modes, SaveConfigMode
scm = {modes.TRAIN: SaveConfigMode(save_interval=100),
        modes.EVAL: SaveConfigMode(save_interval=10)}

hook = TornasoleHook(...,
                     save_config=scm,
                     ...)
```

```
from smdebug.tensorflow import get_collection, modes, SaveConfigMode
get_collection('weights').save_config = {modes.TRAIN: SaveConfigMode(save_interval=10), modes.EVAL: SaveConfigMode(save_interval=1000)}
```

#### Collection
Collection object helps group tensors for easier handling of tensors being saved.
A collection has its own list of tensors, include/exclude regex patterns, reduction config and save config.
This allows setting of different save and reduction configs for different tensors.
These collections are then also available during analysis with `tornasole_rules`.
- Creating or accessing a collection: The following method allows you to access a collection.
It also creates the collection if it does not exist. Here `biases` is the name of the collection.
```
import smdebug.tensorflow as ts
ts.get_collection('biases')
```
- Adding to a collection
```
import smdebug.tensorflow as ts
ts.add_to_collection('inputs', features)
```

- Passing regex pattern to collection
```
import smdebug.tensorflow as ts
ts.get_collection(collection_name).include(['loss*'])
```
Refer [API](api.md) for all methods available when using collections such as setting SaveConfig,
ReductionConfig for a specific collection, retrieving all collections, or resetting all collections.

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


**Examples**
- ```SaveConfig(save_interval=10)``` Saving every 10 steps

- ```SaveConfig(start_step=1000, save_interval=10)``` Save every 10 steps from the 1000th step

- ```SaveConfig(save_steps=[10, 500, 10000, 20000])``` Saves only at the supplied steps

These save config instances can be passed to the hook as follows
```
import smdebug.tensorflow as ts
hook = ts.TornasoleHook(..., save_config=ts.SaveConfig(save_interval=10), ...)
```
Refer [API](api.md) for all parameters available and detailed descriptions for them.

#### ReductionConfig
ReductionConfig allows the saving of certain reductions of tensors instead
of saving the full tensor. By reduction here we mean an operation that converts the tensor to a scalar.
The motivation here is to reduce the amount of data
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
import smdebug.tensorflow as ts
hook = ts.TornasoleHook(..., reduction_config=ts.ReductionConfig(norms=['l1']), ...)
```
Refer [API](api.md) for a full list of the reductions available.

## How to save tensors
There are different ways to save tensors when using smdebug.
Tornasole provides easy ways to save certain standard tensors by way of default collections (a Collection represents a group of tensors).
Examples of such collections are `weights`, `gradients`, `optimizer variables`.
Besides these tensors, you can save tensors by name or regex patterns on those names.
You can also save them by letting Tornasole know which variables in your code are to be saved.
This section will take you through these ways in more detail.

### Default collections
Collection object helps group tensors for easier handling of tensors being saved.
These collections are then also available during analysis.

Tornasole creates a few default collections and populates
them with the relevant tensors.

#### Weights
Weights is a default collection managed by smdebug.
Saving weights is as easy as passing `weights` in the `include_collections` parameter of the hook.
```
import smdebug.tensorflow as ts
hook = ts.TornasoleHook(..., include_collections = ['weights'], ...)
```

#### Gradients
We provide an easy way to populate the collection named `gradients` with the gradients wrt to the weights.
This can be done by wrapping around your optimizer with `wrap_optimizer` as follows.
This will also enable us to access the gradients during analysis without having to identify which tensors out of the saved ones are the gradients.

```
import smdebug.tensorflow as ts
...
opt = hook.wrap_optimizer(opt)
```

You can refer to [customize collections](#customizing-collections) for
information on how you can create the gradients collection manually.

Then, you need to pass `gradients` in the `include_collections` parameter of the hook.
```
import smdebug.tensorflow as ts
hook = ts.TornasoleHook(..., include_collections = ['gradients'], ...)
```

#### Losses
If you are using the default loss functions in Tensorflow, Tornasole can automatically pick up these losses from Tensorflow's losses collection.
In such a case, we only need to specify 'losses' in the `include_collections` argument of the hook.
If you do not pass this argument to the hook, it will save losses by default.
If you are using your custom loss function, you can either add this to Tensorflow's losses collection or Tornasole's losses collection as follows:
```
import smdebug.tensorflow as ts

# if your loss function is not a default TF loss function,
# but is a custom loss function
# then add to the collection losses
loss = ...
ts.add_to_collection('losses', loss)

# specify losses in include_collections
# Note that this is included by default
hook = ts.TornasoleHook(..., include_collections = ['losses'..], ...)
```

#### Optimizer Variables
Optimizer variables such as momentum can also be saved easily with the
above approach of wrapping your optimizer with `wrap_optimizer`
followed by passing `optimizer_variables` in the `include_collections` parameter of the hook.
```
import smdebug.tensorflow as ts
hook = ts.TornasoleHook(..., include_collections = ['optimizer_variables'], ...)
```

Please refer [API](api.md) for more details on using collections

### Customizing collections
You can also create any other customized collection yourself.
You can create new collections as well as modify existing collections
(such as including gradients if you do not want to use the above `wrap_optimizer`)
#### Creating or accessing a collection
Each collection should have a unique name (which is a string).
You can get the collection named as `collection_name` by
calling the following function.
It creates the collection if it does not already exist.
```
ts.get_collection('collection_name')
```
#### Adding tensors
Tensors can be added to a collection by either passing an include regex parameter to the collection.
If you don't know the name of the tensors you want to add, you can also add the tensors to the collection
by the variables representing the tensors in code. The following sections describe these two scenarios.

##### Adding tensors by regex
If you know the name of the tensors you want to save and can write regex
patterns to match those tensornames, you can pass the regex patterns to the collection.
The tensors which match these patterns are included and added to the collection.
```
ts.get_collection('default').include(['foobar/weight*'])
```

**Quick note about names**: TensorFlow layers or operations take a name parameter which along with the name scope
of the layer or variable defines the full name of the operation.
For example, refer [`examples/simple/simple.py`](examples/scripts/simple.py#L20),
the weight there is named `foobar/weight1:0`. Here `foobar/weight1` refers to the
node representing operation in the graph, and the suffix `:0` indicates that this is the 0th output of the node.
To make clear the meaning of a given tensor, it helps to organize your code by name scopes and
set the names of different operations you might be interested in.

##### Adding tensors from variables in the code
If you do not know the names of the tensors you are interested in, you can also just pass the variables to smdebug.
Collection has an add method which takes either a TensorFlow Operation, Variable, or Tensor.

For example, say you want to log the activations of relu layers in your model. You can save them as follows to a
collection named 'relu_activations'. All the tensors represented by this variable (there could be multiple if this line is a loop for instance)
are saved to this collection.
```
x = tf.nn.relu(x)

ts.add_to_collection('relu_activations', x)
```

### Regex pattern
A quick way to save tensors when you know the name of the tensors you want to save and
can write a regex pattern to match those tensornames, is to just pass the regex patterns to the hook.
You can use this approach if you just want to save a small number of tensors and do not care about collections.
The tensors which match these patterns are included and added to the collection named `default`.

```
hook = ts.TornasoleHook(...,
                        include_regex=['foobar/weight*'],
                        ...)
```

**Note** Above does the same as in the Regex section above in Customizing collections.

### Saving all tensors
Tornasole makes it easy to save all the tensors in the model. You just need to set the flag `save_all=True` when creating the hook.
**Please note that this can severely reduce performance of the job and will generate lot of data**

## Analyzing the Results
For full details on how to analyze the tensors saved, go to [DeveloperGuide_Rules](../../rules/DeveloperGuide_Rules.md)

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
