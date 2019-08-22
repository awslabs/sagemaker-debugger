## API

Tornasole PyTorch provides the following constructs:

### Hook
TornasoleHook is the entry point for Tornasole into your program. 


```
class TornasoleHook:

A class used to represent the hook which gets attached to the 
    training process. 

    Attributes
    ----------
    out_dir : str
        represents a path to which the outputs of tornasole will be written to.
        This can be a local path or an S3 prefix of the form s3://bucket_name/prefix.
        Note that for Sagemaker, you always need to specify the out_dir as `/opt/ml/output/tensors`. 
        In the future, we will make this the default in Sagemaker environments.
        
    dry_run : bool
        when dry_run is set to True, behavior is only described in the log file.
        The tensors are not actually saved. 
    
    save_config: SaveConfig object or a dictionary from mode to SaveConfig objects
        SaveConfig allows you to customize when tensors are saved. 
        Hook takes SaveConfig object which is applied as 
        default for all included tensors.
        A collection can optionally have its own SaveConfig object 
        which overrides this for its tensors.
        If you pass a dictionary from mode->SaveConfig, then that
        SaveConfig is applied to tensors included for that mode.
        example: {modes.TRAIN: SaveConfig(save_interval=10), 
                  modes.EVAL:SaveConfig(save_interval=1)}
        Refer to documentation for SaveConfig.
        
    include_regex: list of (str or tensor variables)
        these strings can be regex expressions or simple tensor names.
        if given includes the tensors matched by the expression, 
        or the tensors themselves which were passed.
        if it is empty, does not include any tensor. 
        note that exclude takes precedence over include.
        note also that this is for tensors not in any collection. 
        tensors in collections are handled through 
        include_collections, exclude_collections
        
    reduction_config: ReductionConfig object
        ReductionConfig allows you to save tensors as their reductions 
        instead of saving full tensors. 
        If ReductionConfig is passed then the chosen reductions are applied
        as default for all tensors included.
        A collection can optionally have its own ReductionConfig object
        which overrides this for its tensors.
    include_regex: 
    include_collections: list of str
        takes as input the names of collections which should be saved.
        by default, collections for weights and gradients are created by the hook.
    save_all: bool
        a shortcut for saving all tensors in the model

    def __init__(self,
                 out_dir,
                 dry_run=False,
                 worker=DEFAULT_WORKER_NAME,
                 reduction_config=None,
                 save_config=default_save_config(),
                 include_regex=None,
                 include_collections=['weights', 'bias', 'gradients', 'default'],
                 save_all=False):
```
### Collection

Collection object helps group tensors for easier handling of tensors being saved. 
A collection has its own list of tensors, include/exclude regex patterns, reduction config and save config. 
This allows setting of different save and reduction configs for different tensors. 
These collections are then also available during analysis with `tornasole_rules`. 

#### Creating or accessing a collection
```
import tornasole.pytorch as ts
```

| Function |  Behavior |
|---|---|
| ```ts.get_collection(collection_name)```  |  Returns the collection with the given name. Creates the collection if it doesn't already exist |
| ```ts.get_collections()```  |  Returns all collections as a dictionary with the keys being names of the collections |
| ```ts.add_to_collection(collection_name, args)```  | Equivalent to calling `coll.add(args)` on the collection with name `collection_name` |
| ```ts.add_to_default_collection(args)```  | Equivalent to calling `coll.add(args)` on the collection with the name `default`|
| ```ts.reset_collections()```  | Clears all collections |

#### Methods
The following methods can be called on a collection object.

| Method  |  Behavior |
|---|---|
| ```coll.include(t)```  |  Takes a regex or a list of regex to match tensors to be included to the collection |
| ```coll.add(t)```  | Takes an instance or list or set of tf.Operation/tf.Variable/tf.Tensor to add to the collection  |
| ```coll.get_include_regex()```  | Returns include_regex for the collection  |
| ```coll.get_save_config()```  | Returns save config for the collection  |
| ```coll.set_save_config(s)```  | Sets save config for the collection. You can either pass a SaveConfig instance or a dictionary from mode to SaveConfig |
| ```coll.get_reduction_config()```  | Returns reduction config for the collection  |
| ```coll.set_reduction_config()```  | Sets reduction config for the collection  |
| ```coll.add_module_tensors(module, input=False, output=False)```  | Takes an instance of a module, along with input and output flags. Users can use this Collection to log input/output tensors for a specific module  |




### SaveConfig
SaveConfig class allows you to customize the frequency of saving tensors. 
The hook takes a SaveConfig object which is applied as 
default to all tensors included. A collection can also have its own SaveConfig object which is applied 
to the tensors belonging to that collection.

SaveConfig also allows you to save tensors when certain tensors become nan. 
This list of tensors to watch for is taken as a list of strings representing names of tensors.

```
class SaveConfig:

  """
  
  Attributes
  ----------

  save_interval: int
    save every n steps

  skip_num_steps: int
    start saving after n steps

  save_steps: list of int
    save at all the steps given in this list.
    if this is given, it ignores the save_interval
  """
```  

#### Examples
- ```SaveConfig(save_interval=10)``` Saving every 10 steps

- ```SaveConfig(skip_num_steps=1000, save_interval=10)``` Save every 10 steps after skipping the first 1000 steps

- ```SaveConfig(save_steps=[10, 500, 10000, 20000])``` Saves only at the supplied steps

### ReductionConfig 
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
  Attributes
  ----------

  reductions: list of str
      takes list of names of reductions to be computed.
      should be one of 'min', 'max', 'median', 'mean', 'std', 'variance', 'sum', 'prod'

  abs_reductions: list of str
      takes list of names of reductions to be computed after converting the tensor
      to abs(tensor) i.e. reductions are applied on the absolute values of tensor.
      should be one of 'min', 'max', 'median', 'mean', 'std', 'variance', 'sum', 'prod'

  norms: list of str
      takes names of norms to be computed of the tensor.
      should be one of 'l1', 'l2'

  abs_norms: list of str
        takes names of norms to be computed of the tensor after taking absolute value
        should be one of 'l1', 'l2'
  """
