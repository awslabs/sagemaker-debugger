
# Common API
These objects exist across all frameworks.
- [SageMaker Zero-Code-Change vs. Python API](#sagemaker)
- [Creating a Hook](#creating-a-hook)
    - [Hook from SageMaker](#hook-from-sagemaker)
    - [Hook from Python](#hook-from-python)
- [Modes](#modes)
- [Collection](#collection)
- [SaveConfig](#saveconfig)
- [ReductionConfig](#reductionconfig)
- [Environment Variables](#environment-variables)

## Glossary

The imports assume `import smdebug.{tensorflow,pytorch,mxnet,xgboost} as smd`.

**Hook**: The main interface to use training. This object can be passed as a model hook/callback
in Tensorflow and Keras. It keeps track of collections and writes output files at each step.
- `hook = smd.Hook(out_dir="/tmp/mnist_job")`

**Mode**: One of "train", "eval", "predict", or "global". Helpful for segmenting data based on the phase
you're in. Defaults to "global".
- `train_mode = smd.modes.TRAIN`

**Collection**: A group of tensors. Each collection contains its own save configuration and regexes for
tensors to include/exclude.
- `collection = hook.get_collection("losses")`

**SaveConfig**: A Python dict specifying how often to save losses and tensors.
- `save_config = smd.SaveConfig(save_interval=10)`

**ReductionConfig**: Allows you to save a reduction, such as 'mean' or 'l1 norm', instead of the full tensor.
- `reduction_config = smd.ReductionConfig(reductions=['min', 'max', 'mean'], norms=['l1'])`

**Trial**: The main interface to use when analyzing a completed training job. Access collections and tensors. See [trials documentation](https://link.com).
- `trial = smd.create_trial(out_dir="/tmp/mnist_job")`

**Rule**: A condition that will trigger an exception and terminate the training job early, for example a vanishing gradient. See [rules documentation](https://link.com).


---

## Creating a Hook

### Hook from SageMaker
If you create a SageMaker job and specify the hook configuration in the SageMaker Estimator API
as described in [AWS Docs](https://link.com),
the a JSON file will be automatically written. You can create a hook from this file by calling
```python
hook = smd.{hook_class}.create_from_json_file()
```
with no arguments and then use the hook Python API in your script. `hook_class` will be `Hook` for PyTorch, MXNet, and XGBoost. It will be one of `KerasHook`, `SessionHook`, `EstimatorHook` for TensorFlow.

### Hook from Python
See the framework-specific pages for more details.
* [TensorFlow](https://link.com)
* [PyTorch](https://link.com)
* [MXNet](https://link.com)
* [XGBoost](https://link.com)

---

## Modes
Used to signify which part of training you're in, similar to Keras modes. `GLOBAL` mode is used as
a default. Choose from
```python
smd.modes.TRAIN
smd.modes.EVAL
smd.modes.PREDICT
smd.modes.GLOBAL
```

---

## Collection

The Collection object groups tensors such as "losses", "weights", "biases", or "gradients".
A collection has its own list of tensors, include/exclude regex patterns, reduction config and save config.
This allows setting of different save and reduction configs for different tensors.
These collections are then also available during analysis.

You can choose which of these builtin collections (or define your own) to save in the hook's `include_collections` parameter. By default, only a few collections are saved.

| Framework | include_collections (default) |
|---|---|
| `TensorFlow` | METRICS, LOSSES, SEARCHABLE_SCALARS |
| `PyTorch` | LOSSES, SCALARS |
| `MXNet` | LOSSES, SCALARS |
| `XGBoost` | METRICS |

Each framework has pre-defined settings for certain collections. For example, TensorFlow's KerasHook
will automatically place weights into the `smd.CollectionKeys.WEIGHTS` collection. PyTorch uses the regex
`"^(?!gradient).*weight` to automatically place tensors in the weights collection.

| CollectionKey | Frameworks | Description |
|---|---|---|
| `ALL` | all | Saves all tensors. |
| `DEFAULT` | all | ??? |
| `WEIGHTS` | TensorFlow, PyTorch, MXNet | Matches all weights tensors. |
| `BIASES` | TensorFlow, PyTorch, MXNet | Matches all biases tensors. |
| `GRADIENTS` | TensorFlow, PyTorch, MXNet | Matches all gradients tensors. In TensorFlow non-DLC, must use `hook.wrap_optimizer()`.  |
| `LOSSES` | TensorFlow, PyTorch, MXNet | Matches all loss tensors. |
| `SCALARS` | TensorFlow, PyTorch, MXNet | Matches all scalar tensors, such as loss or accuracy. |
| `METRICS` | TensorFlow, XGBoost | ??? |
| `INPUTS` | TensorFlow | Matches all inputs to a layer (outputs of the previous layer). |
| `OUTPUTS` | TensorFlow | Matches all outputs of a layer (inputs of the following layer). |
| `SEARCHABLE_SCALARS` | TensorFlow | Scalars that will go to SageMaker Metrics. |
| `OPTIMIZER_VARIABLES` | TensorFlow | Matches all optimizer variables. |
| `HYPERPARAMETERS` | XGBoost | ... |
| `PREDICTIONS` | XGBoost | ... |
| `LABELS` | XGBoost | ... |
| `FEATURE_IMPORTANCE` | XGBoost | ... |
| `AVERAGE_SHAP` | XGBoost | ... |
| `FULL_SHAP` | XGBoost | ... |
| `TREES` | XGBoost | ... |




```python
coll = smd.Collection(
    name,
    include_regex = None,
    tensor_names = None,
    reduction_config = None,
    save_config = None,
    save_histogram = True,
)
```
`name` (str): Used to identify the collection.\
`include_regex` (list[str]): The regexes to match tensor names for the collection.\
`tensor_names` (list[str]): A list of tensor names to include.\
`reduction_config`: (ReductionConfig object): Which reductions to store in the collection.\
`save_config` (SaveConfig object): Settings for how often to save the collection.\
`save_histogram` (bool): Whether to save histogram data for the collection. Only used if tensorboard support is enabled. Not computed for scalar collections such as losses.

### Accessing a Collection

| Function |  Behavior |
|---|---|
| ```hook.get_collection(collection_name)```  |  Returns the collection with the given name. Creates the collection with default settings if it doesn't already exist. |
| ```hook.get_collections()```  |  Returns all collections as a dictionary with the keys being names of the collections. |
| ```hook.add_to_collection(collection_name, args)```  | Equivalent to calling `coll.add(args)` on the collection with name `collection_name`. |

### Properties of a Collection
| Property | Description |
|---|---|
| `tensor_names` | Get or set list of tensor names as strings. |
| `include_regex` | Get or set list of regexes to include. |
| `reduction_config` | Get or set the ReductionConfig object. |
| `save_config` | Get or set the SaveConfig object. |


### Methods on a Collection

| Method  |  Behavior |
|---|---|
| ```coll.include(regex)```  |  Takes a regex string or a list of regex strings to match tensors to include in the collection. |
| ```coll.add(tensor)```  | **(TensorFlow only)** Takes an instance or list or set of tf.Tensor/tf.Variable/tf.MirroredVariable/tf.Operation to add to the collection.  |
| ```coll.add_keras_layer(layer, inputs=False, outputs=True)```  | **(tf.keras only)** Takes an instance of a tf.keras layer and logs input/output tensors for that module. By default, only outputs are saved. |
| ```coll.add_module_tensors(module, inputs=False, outputs=True)```  | **(PyTorch only)** Takes an instance of a PyTorch module and logs input/output tensors for that module. By default, only outputs are saved. |
| ```coll.add_block_tensors(block, inputs=False, outputs=True)``` | **(MXNet only)** Takes an instance of a Gluon block,and logs input/output tensors for that module. By default, only outputs are saved. |

---

## SaveConfig
The SaveConfig class customizes the frequency of saving tensors.
The hook takes a SaveConfig object which is applied as default to all tensors included.
A collection can also have a SaveConfig object which is applied to the collection's tensors.

SaveConfig also allows you to save tensors when certain tensors become nan.
This list of tensors to watch for is taken as a list of strings representing names of tensors.

```python
save_config = smd.SaveConfig(
    mode_save_configs = None,
    save_interval = 100,
    start_step = 0,
    end_step = None,
    save_steps = None,
)
```
`mode_save_configs` (dict): Used for advanced cases; see details below.\
`save_interval` (int): How often, in steps, to save tensors. Defaults to 100. \
`start_step` (int): When to start saving tensors.\
`end_step` (int): When to stop saving tensors, exclusive.\
`save_steps` (list[int]): Specific steps to save tensors at. Union with all other parameters.

For example,

`SaveConfig()` will save at steps [0, 100, ...].\
`SaveConfig(save_interval=1)` will save at steps [0, 1, ...]\
`SaveConfig(save_interval=100, end_step=200)` will save at steps [0, 200].\
`SaveConfig(save_interval=100, end_step=201)` will save at steps [0, 100, 200].\
`SaveConfig(save_interval=100, start_step=150)` will save at steps [200, 300, ...].\
`SaveConfig(save_steps=[3, 7])` will save at steps [3, 7].

There is also a more advanced use case, where you specify a different SaveConfig for each mode.
It is best understood through an example:
```python
SaveConfig(mode_save_configs={
    smd.modes.TRAIN: smd.SaveConfigMode(save_interval=1),
    smd.modes.EVAL: smd.SaveConfigMode(save_interval=2),
    smd.modes.PREDICT: smd.SaveConfigMode(save_interval=3),
    smd.modes.GLOBAL: smd.SaveConfigMode(save_interval=4)
})
```
Essentially, create a dictionary mapping modes to SaveConfigMode objects. The SaveConfigMode objects
take the same four parameters (save_interval, start_step, end_step, save_steps) as the main object.
Any mode not specified will default to the default configuration. If a mode is provided but not all
params are specified, we use the default values for non-specified parameters.

---

## ReductionConfig
ReductionConfig allows the saving of certain reductions of tensors instead
of saving the full tensor. The motivation here is to reduce the amount of data
saved, and increase the speed in cases where you don't need the full
tensor. The reduction operations which are computed in the training process
and then saved.

During analysis, these are available as reductions of the original tensor.
Please note that using reduction config means that you will not have
the full tensor available during analysis, so this can restrict what you can do with the tensor saved.
The hook takes a ReductionConfig object which is applied as default to all tensors included.
A collection can also have its own ReductionConfig object which is applied
to the tensors belonging to that collection.

```python
reduction_config = smd.ReductionConfig(
    reductions = None,
    abs_reductions = None,
    norms = None,
    abs_norms = None,
    save_raw_tensor = False,
)
```
`reductions` (list[str]): Takes names of reductions, choosing from "min", "max", "median", "mean", "std", "variance", "sum", "prod".\
`abs_reductions` (list[str]): Same as reductions, except the reduction will be computed on the absolute value of the tensor.\
`norms` (list[str]): Takes names of norms to compute, choosing from "l1", "l2".\
`abs_norms` (list[str]): Same as norms, except the norm will be computed on the absolute value of the tensor.\
`save_raw_tensor` (bool): Saves the tensor directly, in addition to other desired reductions.

For example,

`ReductionConfig(reductions=['std', 'variance'], abs_reductions=['mean'], norms=['l1'])`

will return the standard deviation and variance, the mean of the absolute value, and the l1 norm.


---

## Environment Variables

#### `USE_SMDEBUG`:

Setting this variable to 0 turns off the hook that is created by default. This can be used
if the user doesn't want to use SageMaker Debugger.

#### `SMDEBUG_CONFIG_FILE_PATH`:

Contains the path to the JSON file that describes the smdebug hook.

At the minimum, the JSON config should contain the path where smdebug should output tensors.
Example:

`{ "LocalPath": "/my/smdebug_hook/path" }`

In SageMaker environment, this path is set to point to a pre-defined location containing a valid JSON.
In non-SageMaker environment, SageMaker-Debugger is not used if this environment variable is not set and
a hook is not created manually.

Sample JSON from which a hook can be created:
```json
{
  "LocalPath": "/my/smdebug_hook/path",
  "HookParameters": {
    "save_all": false,
    "include_regex": "regex1,regex2",
    "save_interval": "100",
    "save_steps": "1,2,3,4",
    "start_step": "1",
    "end_step": "1000000",
    "reductions": "min,max,mean"
  },
  "CollectionConfigurations": [
    {
      "CollectionName": "collection_obj_name1",
      "CollectionParameters": {
        "include_regex": "regexe5*",
        "save_interval": 100,
        "save_steps": "1,2,3",
        "start_step": 1,
        "reductions": "min"
      }
    },
  ]
}

```

#### `TENSORBOARD_CONFIG_FILE_PATH`:

Contains the path to the JSON file that specifies where TensorBoard artifacts need to
be placed.

Sample JSON file:

`{ "LocalPath": "/my/tensorboard/path" }`

In SageMaker environment, the presence of this JSON is necessary to log any Tensorboard artifact.
By default, this path is set to point to a pre-defined location in SageMaker.

tensorboard_dir can also be passed while creating the hook [Creating a hook](###Hook from Python) using the API or
in the JSON specified in SMDEBUG_CONFIG_FILE_PATH. For this, export_tensorboard should be set to True.
This option to set tensorboard_dir is available in both, SageMaker and non-SageMaker environments.


#### `CHECKPOINT_CONFIG_FILE_PATH`:

Contains the path to the JSON file that specifies where training checkpoints need to
be placed. This is used in the context of spot training.

Sample JSON file:

`{ "LocalPath": "/my/checkpoint/path" }`

In SageMaker environment, the presence of this JSON is necessary to save checkpoints.
By default, this path is set to point to a pre-defined location in SageMaker.


#### `SAGEMAKER_METRICS_DIRECTORY`:

Contains the path to the directory where metrics will be recorded for consumption by SageMaker Metrics.
This is relevant only in SageMaker environment, where this variable points to a pre-defined location.


#### `TRAINING_END_DELAY_REFRESH`:

During analysis, a [trial](analysis.md) is created to query for tensors from a specified directory. This
directory contains collections, events, and index files. This environment variable
specifies how many seconds to wait before refreshing the index files to check if training has ended
and the tensor is available. By default value, this value is set to 1.


#### `INCOMPLETE_STEP_WAIT_WINDOW`:

During analysis, a [trial](analysis.md) is created to query for tensors from a specified directory. This
directory contains collections, events, and index files. A trial checks to see if a step
specified in the smdebug hook has been completed. This environment variable
specifies the maximum number of incomplete steps that the trial will wait for before marking
half of them as complete. Default: 1000
