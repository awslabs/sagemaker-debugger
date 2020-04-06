
# Saving Tensors API

- [Glossary](#glossary)
- [Hook](#hook)
  - [Creating a Hook](#creating-a-hook)
    - [Hook when using SageMaker Python SDK](#hook-when-using-sagemaker-python-sdk)
    - [Configuring Hook using SageMaker Python SDK](#configuring-hook-using-sagemaker-python-sdk)
    - [Hook from Python constructor](#hook-from-python-constructor)
  - [Common Hook API](#common-hook-api)
  - [TensorFlow specific Hook API](#tensorflow-specific-hook-api)
  - [MXNet specific Hook API](#mxnet-specific-hook-api)
  - [PyTorch specific Hook API](#pytorch-specific-hook-api)
- [Modes](#modes)
- [Collection](#collection)
- [SaveConfig](#saveconfig)
- [ReductionConfig](#reductionconfig)

## Glossary

The imports assume `import smdebug.{tensorflow,pytorch,mxnet,xgboost} as smd`.

**Step**: Step means one the work done by the training job for one batch (i.e. forward and backward pass). (An exception is with TensorFlow's Session interface, where a step also includes the initialization session run calls). SageMaker Debugger is designed in terms of steps. When to save data is specified using steps as well as the invocation of Rules is on a step-by-step basis.

**Hook**: The main class to pass as a callback object, or to create callback functions. It keeps track of collections and writes output files at each step. The current hook implementation does not support merging tensors from current job with tensors from previous job(s). Hence ensure that the 'out_dir' does not exist prior to instantiating the 'Hook' object.
- `hook = smd.Hook(out_dir="/tmp/mnist_job")`

**Mode**: One of "train", "eval", "predict", or "global". Helpful for segmenting data based on the phase
you're in. Defaults to "global".
- `train_mode = smd.modes.TRAIN`

**Collection**: A group of tensors. Each collection contains its configuration for what tensors are part of it, and when to save them.
- `collection = hook.get_collection("losses")`

**SaveConfig**: A Python dict specifying how often to save losses and tensors.
- `save_config = smd.SaveConfig(save_interval=10)`

**ReductionConfig**: Allows you to save a reduction, such as 'mean' or 'l1 norm', instead of the full tensor. Reductions are simple floats.
- `reduction_config = smd.ReductionConfig(reductions=['min', 'max', 'mean'], norms=['l1'])`

**Trial**: The main interface to use when analyzing a completed training job. Access collections and tensors. See [trials documentation](analysis.md).
- `trial = smd.create_trial(out_dir="/tmp/mnist_job")`

**Rule**: A condition to monitor the saved data for. It can trigger an exception when the condition is met, for example a vanishing gradient. See [rules documentation](analysis.md).

---

## Hook
### Creating a Hook
Note that when using Zero Script Change supported containers in SageMaker, you generally do not need to create your hook object except for some advanced use cases where you need access to the hook.

`HookClass` or `hook_class` below will be `Hook` for PyTorch, MXNet, and XGBoost. It will be one of `KerasHook`, `SessionHook` or `EstimatorHook` for TensorFlow.

The framework in `smd` import below refers to one of `tensorflow`, `mxnet`, `pytorch` or `xgboost`.

#### Hook when using SageMaker Python SDK
If you create a SageMaker job and specify the hook configuration in the SageMaker Estimator API
as described in [AWS Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html),
a JSON file containing the hook configuration will be automatically written to the training container. In such a case, you can create a hook from that configuration file by calling
```python
import smdebug.{framework} as smd
hook = smd.{hook_class}.create_from_json_file()
```
with no arguments and then use the hook Python API in your script.

#### Configuring Hook using SageMaker Python SDK
Parameters to the Hook are passed as below when using the SageMaker Python SDK.
```python
from sagemaker.debugger import DebuggerHookConfig
hook_config = DebuggerHookConfig(
    s3_output_path='s3://smdebug-dev-demo-pdx/mnist',
    hook_parameters={
        "parameter": "value"
    })
```
The parameters can be one of the following. The meaning of these parameters will be clear as you review the sections of documentation below. Note that all parameters below have to be strings. So for any parameter which accepts a list (such as save_steps, reductions, include_regex), the value needs to be given as strings separated by a comma between them.
```
dry_run
save_all
include_workers
include_regex
reductions
save_raw_tensor
save_interval
save_steps
start_step
end_step
train.save_interval
train.save_steps
train.start_step
train.end_step
eval.save_interval
eval.save_steps
eval.start_step
eval.end_step
predict.save_interval
predict.save_steps
predict.start_step
predict.end_step
global.save_interval
global.save_steps
global.start_step
global.end_step
```

#### Hook from Python constructor
See the framework-specific pages for more details.

HookClass below can be one of `KerasHook`, `SessionHook`, `EstimatorHook` for TensorFlow, or is just `Hook` for MXNet, Pytorch and XGBoost.

```python
hook = HookClass(
    out_dir,
    export_tensorboard = False,
    tensorboard_dir = None,
    dry_run = False,
    reduction_config = None,
    save_config = None,
    include_regex = None,
    include_collections = None,
    save_all = False,
    include_workers="one"
)
```
##### Arguments
- `out_dir` (str): Path where to save tensors and metadata. This is a required argument. Please ensure that the 'out_dir' does not exist.
- `export_tensorboard` (bool): Whether to export TensorBoard summaries (distributions and histograms for tensors saved, and scalar summaries for scalars saved). Defaults to `False`.  Note that when running on SageMaker this parameter will be ignored. You will need to use the TensorBoardOutputConfig section in API to enable TensorBoard summaries. Refer [SageMaker page](sagemaker.md) for an example.
- `tensorboard_dir` (str): Path where to save TensorBoard artifacts. If this is not passed and `export_tensorboard` is True, then TensorBoard artifacts are saved in `out_dir/tensorboard` . Note that when running on SageMaker this parameter will be ignored. You will need to use the TensorBoardOutputConfig section in API to enable TensorBoard summaries. Refer [SageMaker page](sagemaker.md) for an example.
- `dry_run` (bool): If true, don't write any files
- `reduction_config`: ([ReductionConfig](#reductionconfig) object)  Specifies the reductions to be applied as default for tensors saved. A collection can have its own `ReductionConfig` object which overrides this for the tensors which belong to that collection.
- `save_config`: ([SaveConfig](#saveconfig) object) Specifies when to save tensors.  A collection can have its own `SaveConfig` object which overrides this for the tensors which belong to that collection.
- `include_regex` (list[str]): list of regex patterns which specify the tensors to save. Tensors whose names match these patterns will be saved
- `include_collections` (list[str]): List of which collections to save specified by name
- `save_all` (bool): Saves all tensors and collections. Increases the amount of disk space used, and can reduce the performance of the training job significantly, depending on the size of the model.
- `include_workers` (str): Used for distributed training. It can take the values `one` or `all`. `one` means only the tensors from one chosen worker will be saved. This is the default behavior. `all` means tensors from all workers will be saved.

### Common Hook API
These methods are common for all hooks in any framework.

Note that `smd` import below translates to `import smdebug.{framework} as smd`.

| Method | Arguments | Behavior |
| --- | --- | --- |
|`add_collection(collection)` | `collection (smd.Collection)` | Takes a Collection object and adds it to the CollectionManager that the Hook holds. Note that you should only pass in a Collection object for the same framework as the hook |
|`get_collection(name)`| `name (str)` | Returns collection identified by the given name |
|`get_collections()` | - | Returns all collection objects held by the hook |
|`set_mode(mode)`| value of the enum `smd.modes` | Sets mode of the job, can be one of `smd.modes.TRAIN`, `smd.modes.EVAL`, `smd.modes.PREDICT` or `smd.modes.GLOBAL`. Refer [Modes](#modes) for more on that. |
|`create_from_json_file(`<br/>`  json_file_path=None)` | `json_file_path (str)` | Takes the path of a file which holds the json configuration of the hook, and creates hook from that configuration. This is an optional parameter. <br/> If this is not passed it tries to get the file path from the value of the environment variable `SMDEBUG_CONFIG_FILE_PATH` and defaults to `/opt/ml/input/config/debughookconfig.json`. When training on SageMaker you do not have to specify any path because this is the default path that SageMaker writes the hook configuration to.
|`close()` | - | Closes all files that are currently open by the hook |
| `save_scalar(`<br/>`name, `<br/>`value, `<br/>`sm_metric=False)` | `name (str)` <br/> `value (float)` <br/> `sm_metric (bool)`| Saves a scalar value by the given name. Passing `sm_metric=True` flag also makes this scalar available as a SageMaker Metric to show up in SageMaker Studio. Note that when `sm_metric` is False, this scalar always resides only in your AWS account, but setting it to True saves the scalar also on AWS servers. The default value of `sm_metric` for this method is False. |

### TensorFlow specific Hook API
Note that there are three types of Hooks in TensorFlow: SessionHook, EstimatorHook and KerasHook based on the TensorFlow interface being used for training. [This page](tensorflow.md) shows examples of each of these.

| Method | Arguments | Returns | Behavior |
| --- | --- | --- | --- |
| `wrap_optimizer(optimizer)` | `optimizer` (tf.train.Optimizer or tf.keras.Optimizer) | Returns the same optimizer object passed with a couple of identifying markers to help `smdebug`. This returned optimizer should be used for training. | When not using Zero Script Change environments, calling this method on your optimizer is necessary for SageMaker Debugger to identify and save gradient tensors. Note that this method returns the same optimizer object passed and does not change your optimization logic. If the hook is of type `KerasHook`, you can pass in either an object of type `tf.train.Optimizer` or `tf.keras.Optimizer`. If the hook is of type `SessionHook` or `EstimatorHook`, the optimizer can only be of type `tf.train.Optimizer`.  This new
| `add_to_collection(`<br/>  `collection_name, variable)` | `collection_name (str)` : name of the collection to add to. <br/> `variable` parameter to pass to the collection's `add` method. | `None` | Calls the `add` method of a collection object. See [this section](#collection) for more. |

APIs specific to training scripts using TF 2.x GradientTape ([Example](tensorflow.md#TF 2.x GradientTape example)):

| Method | Arguments | Returns | Behavior |
| --- | --- | --- | --- |
| `wrap_tape(tape)` | `tape` (tensorflow.python.eager.backprop.GradientTape) | Returns a tape object with three identifying markers to help `smdebug`. This returned tape should be used for training. | When not using Zero Script Change environments, calling this method on your tape is necessary for SageMaker Debugger to identify and save gradient tensors. Note that this method returns the same tape object passed.
| `record_tensor_value(`<br/>  `tensor_name, tensor_value)` | `tensor_name (str)` : name of the tensor to save. <br/> `tensor_value` EagerTensor to save. | `None` | Manually save metrics tensors while using TF 2.x GradientTape. |

### MXNet specific Hook API

| Method | Arguments | Behavior |
| --- | --- | --- |
| `register_block(block)` | `block (mx.gluon.Block)` | Calling this method applies the hook to the Gluon block representing the model, so SageMaker Debugger gets called by MXNet and can save the tensors required. |

### PyTorch specific Hook API


| Method | Arguments | Behavior |
| --- | --- | --- |
| `register_module(module)` | `module (torch.nn.Module)` | Calling this method applies the hook to the Torch Module representing the model, so SageMaker Debugger gets called by PyTorch and can save the tensors required. |
| `register_loss(loss_module)` | `loss_module (torch.nn.modules.loss._Loss)` | Calling this method applies the hook to the Torch Module representing the loss, so SageMaker Debugger can save losses |

---

## Modes
Used to signify which part of training you're in, similar to Keras modes. `GLOBAL` mode is used as
a default when no mode was set. Choose from
```python
smdebug.modes.TRAIN
smdebug.modes.EVAL
smdebug.modes.PREDICT
smdebug.modes.GLOBAL
```

The modes enum is also available under the alias `smdebug.{framework}.modes`.

---

## Collection

The construct of a Collection groups tensors together. A Collection is identified by a string representing the name of the collection. It can be used to group tensors of a particular kind such as "losses", "weights", "biases", or "gradients". A Collection has its own list of tensors specified by include regex patterns, and other parameters determining how these tensors should be saved and when. Using collections enables you to save different types of tensors at different frequencies and in different forms. These collections are then also available during analysis so you can query a group of tensors at once.

There are a number of built-in collections that SageMaker Debugger manages by default. This means that the library takes care of identifying what tensors should be saved as part of that collection. You can also define custom collections, to do which there are couple of different ways.

You can specify which of these collections to save in the hook's `include_collections` parameter, or through the `collection_configs` parameter to the `DebuggerHookConfig` in the SageMaker Python SDK.

### Built in Collections
Below is a comprehensive list of the built-in collections that are managed by SageMaker Debugger. The Hook identifes the tensors that should be saved as part of that collection for that framework and saves them if they were requested.

The names of these collections are all lower case strings.

| Name | Supported by frameworks/hooks | Description |
|---|---|---|
| `all` | all | Matches all tensors |
| `default` | all | It's a default collection created, which matches the regex patterns passed as `include_regex` to the Hook |
| `weights` | TensorFlow, PyTorch, MXNet | Matches all weights of the model |
| `biases` | TensorFlow, PyTorch, MXNet | Matches all biases of the model |
| `gradients` | TensorFlow, PyTorch, MXNet | Matches all gradients of the model. In TensorFlow when not using Zero Script Change environments, must use `hook.wrap_optimizer()`.  |
| `losses` | TensorFlow, PyTorch, MXNet | Saves the loss for the model |
| `metrics` | TensorFlow's KerasHook, XGBoost | For KerasHook, saves the metrics computed by Keras for the model. For XGBoost, the evaluation metrics computed by the algorithm. |
| `outputs` | TensorFlow's KerasHook | Matches the outputs of the model |
| `sm_metrics` | TensorFlow | You can add scalars that you want to show up in SageMaker Metrics to this collection. SageMaker Debugger will save these scalars both to the out_dir of the hook, as well as to SageMaker Metric. Note that the scalars passed here will be saved on AWS servers outside of your AWS account. |
| `optimizer_variables` | TensorFlow's KerasHook | Matches all optimizer variables, currently only supported in Keras. |
| `hyperparameters` | XGBoost | [Booster paramameters](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html) |
| `predictions` | XGBoost | Predictions on validation set (if provided) |
| `labels` | XGBoost | Labels on validation set (if provided) |
| `feature_importance` | XGBoost | Feature importance given by [get_score()](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score) |
| `full_shap` | XGBoost | A matrix of (nsmaple, nfeatures + 1) with each record indicating the feature contributions ([SHAP values](https://github.com/slundberg/shap)) for that prediction. Computed on training data with [predict()](https://github.com/slundberg/shap) |
| `average_shap` | XGBoost | The sum of SHAP value magnitudes over all samples. Represents the impact each feature has on the model output. |
| `trees` | XGBoost | Boosted tree model given by [trees_to_dataframe()](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.trees_to_dataframe) |

### Default collections saved
The following collections are saved regardless of the hook configuration.

| Framework | Default collections saved |
|---|---|
| `TensorFlow` | METRICS, LOSSES, SM_METRICS |
| `PyTorch` | LOSSES |
| `MXNet` | LOSSES |
| `XGBoost` | METRICS |


 If for some reason, you want to disable the saving of these collections, you can do so by setting end_step to 0 in the collection's SaveConfig.
 When using the SageMaker Python SDK this would look like
 ```python
from sagemaker.debugger import DebuggerHookConfig, CollectionConfig
hook_config = DebuggerHookConfig(
    s3_output_path='s3://smdebug-dev-demo-pdx/mnist',
    collection_configs=[
        CollectionConfig(name="metrics", parameters={"end_step": 0})
    ]
)
 ```
 When configuring the Collection in your Python script, it would be as follows:
 ```python
 hook.get_collection("metrics").save_config.end_step = 0
 ```

### Creating or retrieving a Collection

| Function |  Behavior |
|---|---|
| ```hook.get_collection(collection_name)```  |  Returns the collection with the given name. Creates the collection with default configuration if it doesn't already exist. A new collection created by default does not match any tensor and is configured to save histograms and distributions along with the tensor if tensorboard support is enabled, and uses the reduction configuration and save configuration passed to the hook. |

### Properties of a Collection
| Property | Description |
|---|---|
| `tensor_names` | Get or set list of tensor names as strings |
| `include_regex` | Get or set list of regexes to include. Tensors whose names match these regex patterns will be included in the collection |
| `reduction_config` | Get or set the ReductionConfig object to be used for tensors part of this collection |
| `save_config` | Get or set the SaveConfig object to be used for tensors part of this collection |
| `save_histogram` | Get or set the boolean flag which determines whether to write histograms to enable histograms and distributions in TensorBoard, for tensors part of this collection. Only applicable if TensorBoard support is enabled.|


### Methods on a Collection

| Method  |  Behavior |
|---|---|
| ```coll.include(regex)```  |  Takes a regex string or a list of regex strings to match tensors to include in the collection. |
| ```coll.add(tensor)```  | **(TensorFlow only)** Takes an instance or list or set of tf.Tensor/tf.Variable/tf.MirroredVariable/tf.Operation to add to the collection.  |
| ```coll.add_keras_layer(layer, inputs=False, outputs=True)```  | **(tf.keras only)** Takes an instance of a tf.keras layer and logs input/output tensors for that module. By default, only outputs are saved. |
| ```coll.add_module_tensors(module, inputs=False, outputs=True)```  | **(PyTorch only)** Takes an instance of a PyTorch module and logs input/output tensors for that module. By default, only outputs are saved. |
| ```coll.add_block_tensors(block, inputs=False, outputs=True)``` | **(MXNet only)** Takes an instance of a Gluon block,and logs input/output tensors for that module. By default, only outputs are saved. |

### Configuring Collection using SageMaker Python SDK
Parameters to configure Collection are passed as below when using the SageMaker Python SDK.
```python
from sagemaker.debugger import CollectionConfig
coll_config = CollectionConfig(
    name="weights",
    parameters={ "parameter": "value" })
```
The parameters can be one of the following. The meaning of these parameters will be clear as you review the sections of documentation below. Note that all parameters below have to be strings. So any parameter which accepts a list (such as save_steps, reductions, include_regex), needs to be given as strings separated by a comma between them.

```
include_regex
save_histogram
reductions
save_raw_tensor
save_interval
save_steps
start_step
end_step
train.save_interval
train.save_steps
train.start_step
train.end_step
eval.save_interval
eval.save_steps
eval.start_step
eval.end_step
predict.save_interval
predict.save_steps
predict.start_step
predict.end_step
global.save_interval
global.save_steps
global.start_step
global.end_step
```


---

## SaveConfig
The SaveConfig class customizes the frequency of saving tensors.
The hook takes a SaveConfig object which is applied as default to all tensors included.
A collection can also have a SaveConfig object which is applied to the collection's tensors.
You can also choose to have different configuration for when to save tensors based on the mode of the job.

This class is available in the following namespaces `smdebug` and `smdebug.{framework}`.

```python
import smdebug as smd
save_config = smd.SaveConfig(
    mode_save_configs = None,
    save_interval = 100,
    start_step = 0,
    end_step = None,
    save_steps = None,
)
```
##### Arguments
- `mode_save_configs` (dict): Used for advanced cases; see details below.
- `save_interval` (int): How often, in steps, to save tensors. Defaults to 500. A step is saved if `step % save_interval == 0`
- `start_step` (int): When to start saving tensors.
- `end_step` (int): When to stop saving tensors, exclusive.
- `save_steps` (list[int]): Specific steps to save tensors at. Union with save_interval.

##### Examples

- `SaveConfig()` will save at steps 0, 500, ...
- `SaveConfig(save_interval=1)` will save at steps 0, 1, ...
- `SaveConfig(save_interval=100, end_step=200)` will save at steps 0, 100
- `SaveConfig(save_interval=100, end_step=201)` will save at steps 0, 100, 200
- `SaveConfig(save_interval=100, start_step=150)` will save at steps 200, 300, ...
- `SaveConfig(save_steps=[3, 7])` will save at steps 0, 3, 7, 500, ...

##### Specifying different configuration based on mode
There is also a more advanced use case, where you specify a different SaveConfig for each mode.
It is best understood through an example:
```python
import smdebug as smd
smd.SaveConfig(mode_save_configs={
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

#### Configuration using SageMaker Python SDK
Refer [Configuring Hook using SageMaker Python SDK](#configuring-hook-using-sagemaker-python-sdk) and [Configuring Collection using SageMaker Python SDK](#configuring-collection-using-sagemaker-python-sdk)

---

## ReductionConfig
ReductionConfig allows the saving of certain reductions of tensors instead
of saving the full tensor. The motivation here is to reduce the amount of data
saved, and increase the speed in cases where you don't need the full
tensor. The reduction operations which are computed in the training process
and then saved.

During analysis, these are available as reductions of the original tensor.
Please note that using reduction config means that you will not have
the full tensor available during analysis, so this can restrict what you can do with the tensor saved. You can choose to also save the raw tensor along with the reductions if you so desire.

The hook takes a ReductionConfig object which is applied as default to all tensors included.
A collection can also have its own ReductionConfig object which is applied
to the tensors belonging to that collection.

```python
import smdebug as smd
reduction_config = smd.ReductionConfig(
    reductions = None,
    abs_reductions = None,
    norms = None,
    abs_norms = None,
    save_raw_tensor = False,
)
```

##### Arguments
- `reductions` (list[str]): Takes names of reductions, choosing from "min", "max", "median", "mean", "std", "variance", "sum", "prod"
- `abs_reductions` (list[str]): Same as reductions, except the reduction will be computed on the absolute value of the tensor
- `norms` (list[str]): Takes names of norms to compute, choosing from "l1", "l2"
- `abs_norms` (list[str]): Same as norms, except the norm will be computed on the absolute value of the tensor
- `save_raw_tensor` (bool): Saves the tensor directly, in addition to other desired reductions

For example,

`ReductionConfig(reductions=['std', 'variance'], abs_reductions=['mean'], norms=['l1'])`

will save the standard deviation and variance, the mean of the absolute value, and the l1 norm.

#### Configuration using SageMaker Python SDK
The reductions are passed as part of the "reductions" parameter to HookParameters or Collection Parameters.
Refer [Configuring Hook using SageMaker Python SDK](#configuring-hook-using-sagemaker-python-sdk) and [Configuring Collection using SageMaker Python SDK](#configuring-collection-using-sagemaker-python-sdk) for more on that.

The parameter "reductions" can take a comma separated string consisting of the following values:
```
min
max
median
mean
std
variance
sum
prod
l1
l2
abs_min
abs_max
abs_median
abs_mean
abs_std
abs_variance
abs_sum
abs_prod
abs_l1
abs_l2
```

---

## Frameworks

For details on what's supported for different framework, go here:
* [TensorFlow](tensorflow.md)
* [PyTorch](pytorch.md)
* [MXNet](mxnet.md)
* [XGBoost](xgboost.md)
