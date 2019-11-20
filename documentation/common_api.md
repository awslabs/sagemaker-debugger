
# Common API
These objects exist across all frameworks.
- [Collection](#collection)
- [SaveConfig](#saveconfig)
- [ReductionConfig](#reductionconfig)



## Collection

The Collection object joins tensors into groups such as "losses", "weights", "biases", or "gradients".
A collection has its own list of tensors, include/exclude regex patterns, reduction config and save config.
This allows setting of different save and reduction configs for different tensors.
These collections are then also available during analysis.

### Creating/Accessing a Collection

| Function |  Behavior |
|---|---|
| ```hook.get_collection(collection_name)```  |  Returns the collection with the given name. Creates the collection if it doesn't already exist. |
| ```hook.get_collections()```  |  Returns all collections as a dictionary with the keys being names of the collections. |
| ```hook.add_to_collection(collection_name, args)```  | Equivalent to calling `coll.add(args)` on the collection with name `collection_name`. |
| ```hook.add_to_default_collection(args)```  | Equivalent to calling `coll.add(args)` on the collection with the name `default`. |

### Methods on a Collection

| Method  |  Behavior |
|---|---|
| ```coll.include(regex)```  |  Takes a regex string or a list of regex strings to match tensors to include in the collection. |
| ```coll.add(tensor)```  | Takes an instance or list or set of tf.Operation/tf.Variable/tf.Tensor to add to the collection.  |
| ```coll.include_regex```  | Get or set include_regex for the collection.  |
| ```coll.save_config```  | Get or set save_config for the collection.  |
| ```coll.reduction_config```  | Get or set reduction config for the collection.  |
| ```coll.add_module_tensors(module, inputs=False, outputs=True)```  | **(PyTorch only)** Takes an instance of a PyTorch module and logs input/output tensors for that module. By default, only outputs are saved. |
| ```coll.add_block_tensors(block, inputs=False, outputs=True)``` | **(MXNet only)** Takes an instance of a Gluon block,and logs input/output tensors for that module. By default, only outputs are saved. |



## SaveConfig
The SaveConfig class customizes the frequency of saving tensors.
The hook takes a SaveConfig object which is applied as default to all tensors included.
A collection can also have a SaveConfig object which is applied to the collection's tensors.

SaveConfig also allows you to save tensors when certain tensors become nan.
This list of tensors to watch for is taken as a list of strings representing names of tensors.

```
def __init__(
    mode_save_configs = None,
    save_interval = 100,
    start_step = 0,
    end_step = None,
    save_steps = None,
)
```
`mode_save_configs` (dict): Used for advanced cases; see details below.\
`save_interval` (int): How often, in steps, to save tensors.\
`start_step` (int): When to start saving tensors.\
`end_step` (int): When to stop saving tensors, exclusive.\
`save_steps` (list[int]): Specific steps to save tensors at. Overrides all other parameters.

For example,

`SaveConfig()` will save at steps [0, 100, ...].\
`SaveConfig(save_interval=1)` will save at steps [0, 1, ...]\
`SaveConfig(save_interval=100, end_step=200)` will save at steps [0, 200].\
`SaveConfig(save_interval=100, end_step=201)` will save at steps [0, 100, 200].\
`SaveConfig(save_interval=100, start_step=150)` will save at steps [200, 300, ...].\
`SaveConfig(save_steps=[3, 7])` will save at steps [3, 7].

There is also a more advanced use case, where you specify a different SaveConfig for each mode.
It is best understood through an example:
```
SaveConfig(mode_save_configs={
    smd.modes.TRAIN: smd.SaveConfigMode(save_interval=1),
    smd.modes.EVAL: smd.SaveConfigMode(save_interval=2),
    smd.modes.PREDICT: smd.SaveConfigMode(save_interval=3),
    smd.modes.GLOBAL: smd.SaveConfigMode(save_interval=4)
})
```
Essentially, create a dictionary mapping modes to SaveConfigMode objects. The SaveConfigMode objects
take the same four parameters (save_interval, start_step, end_step, save_steps) as the main object.
Any mode not specified will default to the default configuration.

## ReductionConfig
ReductionConfig allows the saving of certain reductions of tensors instead
of saving the full tensor. The motivation here is to reduce the amount of data
saved, and increase the speed in cases where you don't need the full
tensor.  The reduction operations which are computed in the training process
and then saved.

During analysis, these are available as reductions of the original tensor.
Please note that using reduction config means that you will not have
the full tensor available during analysis, so this can restrict what you can do with the tensor saved.
The hook takes a ReductionConfig object which is applied as default to all tensors included.
A collection can also have its own ReductionConfig object which is applied
to the tensors belonging to that collection.

```
def __init__(
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
`abs_norms` (list[str]): Same as norms, except the norm will be computed on the absolute value of the tensor.

For example,

`ReductionConfig(reductions=['std', 'variance'], abs_reductions=['mean'], norms=['l1'])`

will return the standard deviation and variance, the mean of the absolute value, and the l1 norm.
