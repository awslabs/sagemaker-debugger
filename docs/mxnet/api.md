## API

Tornasole MXNet provides the following constructs:
### Hook
TornasoleHook is the entry point for Tornasole into your program.

```

	class TornasoleHook
    """
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

    worker: str
        name of worker in a multi process training job
        outputs and tensors are organized by this name during retrieval.

    save_config: SaveConfig object or a dictionary from mode to SaveConfigMode objects
        SaveConfig allows you to customize when tensors are saved.
        Hook takes SaveConfig object which is applied as
        default for all included tensors.
        A collection can optionally have its own SaveConfig object
        which overrides this for its tensors.
        If you pass a dictionary from mode->SaveConfig, then that
        SaveConfig is applied to tensors included for that mode.
        example: {modes.TRAIN: SaveConfigMode(save_interval=10),
                  modes.EVAL:SaveConfigMode(save_interval=1)}
        Refer to documentation for SaveConfig.

    reduction_config: ReductionConfig object
        ReductionConfig allows you to save tensors as their reductions
        instead of saving full tensors.
        If ReductionConfig is passed then the chosen reductions are applied
        as default for all tensors included.
        A collection can optionally have its own ReductionConfig object
        which overrides this for its tensors.

    include_regex: list of str
        takes as input the list of string representing regular expressions. Tensors whose names match
        these regular expressions will be saved. These tensors will be available as part of the `default`
        collection.

    include_collections: list of str
        takes as input the names of collections which should be saved.
        by default, ['weights','gradients', 'biases', 'default'] are passed to include_collections.

    save_all: bool
        a shortcut for saving all tensors in the model.
        tensors are all grouped into the `default` collection

    def __init__(self,
        out_dir,
        dry_run=False,
        worker='worker0',
        reduction_config=None,
        save_config=SaveConfig(save_interval=100),
        include_regex=None,
        include_collections=['weights', 'gradients', 'biases', 'default'],
        save_all=False,
        ):
```


It also has an important method which can be used to set the appropriate mode.
Modes can refer to 'training', 'evaluation' or 'prediction'. They can be set as follows:

```hook.set_mode(ts.modes.TRAIN)```,

```hook.set_mode(ts.modes.EVAL)``` or

```hook.set_mode(ts.modes.PREDICT)```.

This allows you to group steps by mode which allows for clearer analysis. Tornasole
also allows you to see a global ordering of steps which makes it clear after how many training
steps did a particular evaluation step happen. If you do not set this mode, all steps are saved under
a `default` mode.

The _save\_config_ parameter is optional. If not specified, the TornasoleHook will use a default SaveConfig that stores tensors with step_interval=100. That is, the  tensors will be saved every 100th step.

The _reduction\_config_ is optional. If not specified, the reductions are not applied to the stored tensors.


### Collection

Collection object helps group tensors for easier handling of tensors being saved.
A collection has its own list of tensors, include/exclude regex patterns, reduction config and save config.
This allows setting of different save and reduction configs for different tensors.
These collections are then also available during analysis with `tornasole_rules`.

#### Creating or accessing a collection

```
import tornasole.mxnet as ts
```


| Function |  Behavior |
|----|----|
| ```ts.get_collection(collection_name)```  |  Returns the collection with the given name. Creates the collection if it doesn't already exist |
| ```ts.get_collections()```  |  Returns all collections as a dictionary with the keys being names of the collections |
| ```ts.add_to_collection(collection_name, args)```  | Equivalent to calling `coll.add(args)` on the collection with name `collection_name` |
| ```ts.add_to_default_collection(args)```  | Equivalent to calling `coll.add(args)` on the collection with the name `default`|
| ```ts.reset_collections()```  | Clears all collections |

#### Methods
The following methods can be called on a collection object.


| Method  |  Behavior |
|----|----|
| ```coll.include(t)```  |  Takes a regex or a list of regex to match tensors to be included to the collection |
| ```coll.add_block_tensors(block, input=False, output=False)```  | Takes an instance Gluon block, input and output flags. Users can use this Collection to log input/output tensors for a specific block  |
| ```coll.include_regex```  | Gets include_regex for the collection  |
| ```coll.save_config```  | Get or set save config for the collection. You can either pass a SaveConfig instance or a dictionary from mode to SaveConfigMode |
| ```coll.reduction_config```  | Get or set reduction config for the collection  |

### SaveConfig
SaveConfig class allows you to customize the frequency of saving tensors.
The hook takes a SaveConfig object which is applied as
default to all tensors included.
A collection can also have its own SaveConfig object which is applied
to the tensors belonging to that collection.

SaveConfig also allows you to save tensors when certain tensors become nan.
This list of tensors to watch for is taken as a list of strings representing names of tensors.

```

    class SaveConfig:

    Attributes
    ----------

    save_interval: int
    allows you to save every n steps by passing n to save_interval

    start_step: int
            Allows you to start saving from a given step, defaults to 0

    end_step: int
        allows you to save till a given step. Excludes this end_step
        defaults to None, i.e. till end of job

    save_steps: list of int
        save at all the steps given in this list.
        if this is given, it ignores the save_interval.
```

The default value of _save\_interval_ is 100. The TornasoleHook that uses a default SaveConfig object will store the tensors every 100th step.


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
      should be one of 'min', 'max', 'median', 'mean', 'std', 'sum', 'prod'

    abs_reductions: list of str
      takes list of names of reductions to be computed after converting the tensor
      to abs(tensor) i.e. reductions are applied on the absolute values of tensor.
      should be one of 'min', 'max', 'median', 'mean', 'std', 'sum', 'prod'

    norms: list of str
      takes names of norms to be computed of the tensor.
      should be one of 'l1', 'l2'

    abs_norms: list of str
        takes names of norms to be computed of the tensor after taking absolute value
        should be one of 'l1', 'l2'
```
