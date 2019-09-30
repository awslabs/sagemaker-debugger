## API

Tornasole XGBoost provides the following constructs:

### Hook
TornasoleHook is the entry point for Tornasole into your program.

```
def __init__(
        self,
        out_dir: str,
        dry_run: bool = False,
        worker: str = TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
        reduction_config=None,
        save_config: Optional[SaveConfig] = None,
        include_regex: Optional[List[str]] = None,
        include_collections: Union[None, List[str], List[Collection]] = None,
        save_all: bool = False,
        shap_data: Union[None, Tuple[str, str], DMatrix] = None
        ) -> None:
    """
    This class represents the hook which is meant to be used a callback
    function in XGBoost.

    Example
    -------
    >>> from tornasole.xgboost import TornasoleHook
    >>> tornasole_hook = TornasoleHook()
    >>> xgboost.train(prams, dtrain, callbacks=[tornasole_hook])

    Parameters
    ----------
    out_dir: A path into which tornasole outputs will be written.
    dry_run: When dry_run is True, behavior is only described in the log
        file, and evaluations are not actually saved.
    worker: name of worker in distributed setting.
    reduction_config: This parameter is not used.
        Placeholder to keep the API consistent with other hooks.
    save_config: A tornasole_core.SaveConfig object.
        See an example at https://github.com/awslabs/tornasole_core/blob/master/tests/test_save_config.py
    include_regex: Tensors matching these regular expressions will be
        available as part of the 'default' collection.
    include_collections: Tensors that should be saved.
        If not given, all known collections will be saved.
    save_all: If true, all evaluations are saved in the collection 'all'.
    shap_data: When this parameter is a tuple (file path, content type) or
        an xboost.DMatrix instance, the average feature contributions
        (SHAP values) will be calcaulted against the provided data set.
        content type can be either 'csv' or 'libsvm', e.g.,
        shap_data = ('/path/to/train/file', 'csv') or
        shap_data = ('/path/to/validation/file', 'libsvm') or
        shap_data = xgboost.DMatrix('train.svm.txt')
    """
```

The `save_config` parameter is optional. If not specified, the TornasoleHook
will use a default SaveConfig that stores tensors with `step_interval`=100.
That is, the tensors will be saved every 100th step.

The `reduction_config` is not supported for XGBoost. If specified, its value
will be ignored.

### Collection

Collection object helps group tensors for easier handling of tensors being saved.
A collection has its own list of tensors, include/exclude regex patterns, and save config.
This allows setting of different save configs for different tensors.
These collections are then also available during analysis with `tornasole_rules`.

#### Default Collections
Currently, the XGBoost TornasoleHook creates Collection objects for
'metric', 'feature\_importance', 'average\_shap', and 'default'.

##### Evaluation metrics: metric
When the [eval\_metric](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
parameter is specified in `params` or the [eval](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train)
parameter is set in [xgboost.train()](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train),
XGBoost computes the evaluatoin metrics for training and/or validation data.
TornasoleHook will match the regex pattern of these evaluation metrics
and provide them in a collection named `metric`.

##### Feature importances: feature\_importance
Tornasole provides the feature importance of each feature in a collection named
`feature_importance`.
The feature importance is defined as the number of times a feature is used to
split the data across all trees.
These values are equivalent to the values you would get from
[xgboost.Booster.get\_score](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score).
Zero-valued feature importances are not included in the collection.

#### SHAP values: average\_shap
If you use the `shap_data` parameter in `tornasole.xgboost.TornasoleHook`,
Tornasole provides the average [SHAP](https://github.com/slundberg/shap) value
of each feature in a collection named `average_shap`.
The `shap_data` parameter can be a tuple (file path, content type) or an
[xboost.DMatrix](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.DMatrix)
instance.
Zero-valued SHAP values are not included in the collection.

collections contain the regex pattern that match with tensors of type
evaluation metrics, feature importances, and SHAP values. The regex pattern for
the 'default' collection is set when user specifies *include\_regex* with
TornasoleHook or sets the *save_all=True*.  These collections use the SaveConfig
parameter provided with the TornasoleHook initialization. The TornasoleHook
will store the related tensors, if user does not specify any special collection
with *include\_collections* parameter. If user specifies a collection with
*include\_collections* the above default collections will not be in effect.

#### Creating or accessing a collection

```
import tornasole.xgboost as tx
```

| Function |  Behavior |
|----|----|
| ```tx.get_collection(collection_name)```  |  Returns the collection with the given name. Creates the collection if it doesn't already exist |
| ```tx.get_collections()```  |  Returns all collections as a dictionary with the keys being names of the collections |
| ```tx.add_to_collection(collection_name, args)```  | Equivalent to calling `coll.add(args)` on the collection with name `collection_name` |
| ```tx.add_to_default_collection(args)```  | Equivalent to calling `coll.add(args)` on the collection with the name `default`|
| ```tx.reset_collections()```  | Clears all collections |

#### Methods

The following methods can be called on a collection object.


| Method  |  Behavior |
|----|----|
| ```coll.include(t)```  |  Takes a regex or a list of regex to match tensors to be included to the collection |
| ```coll.get_include_regex()```  | Returns include_regex for the collection  |
| ```coll.get_save_config()```  | Returns save config for the collection  |
| ```coll.set_save_config(s)```  | Sets save config for the collection. You can either pass a SaveConfig instance or a dictionary from mode to SaveConfig |

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

ReductionConfig is not currently used in XGBoost Tornasole.
When Tornasole is used with deep learning frameworks, such as MXNet,
Tensorflow, or PyTorch, ReductionConfig allows the saving of certain
reductions of tensors instead of saving the full tensor.
By reduction here we mean an operation that converts the tensor to a scalar.
However, in XGBoost, we currently support evaluation metrics, feature
importances, and average SHAP values, which are all scalars and not tensors.
Therefore, if the `reduction_config` parameter is set in
`tornasole.xgboost.TornasoleHook`, it will be ignored and not used at all.
