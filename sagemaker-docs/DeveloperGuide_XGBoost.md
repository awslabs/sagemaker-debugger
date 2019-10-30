# Tornasole for XGBoost

Tornasole is a new capability of Amazon SageMaker that allows debugging machine learning training. Tornasole helps you to monitor your training in near real time using rules and would provide you alerts, once it has detected inconsistency in training.

## Quickstart

If you want to quickly run an end-to-end example, please refer to [XGBoost notebook example](examples/notebooks/xgboost.ipynb) to see tornasole working.

Integrating Tornasole into the training job can be accomplished by following steps below.

### Import the Tornasole package

Import the TornasoleHook class along with other helper classes in your training script as shown below

```
from tornasole.xgboost import TornasoleHook
from tornasole import SaveConfig
```

### Instantiate and initialize tornasole hook

```
    # Create SaveConfig that instructs engine to log graph tensors every 10 steps.
    save_config = SaveConfig(save_interval=10)
    # Create a hook that logs evaluation metrics and feature importances while training the model.
    hook = TornasoleHook(save_config=save_config)
```

Using the *Collection* object and/or *include\_regex* parameter of TornasoleHook , users can control which tensors will be stored by the TornasoleHook.
The section [How to save tensors](#how-to-save-tensors) explains various ways users can create *Collection* object to store the required tensors.

The *SaveConfig* object controls when these tensors are stored. The tensors can be stored for specific steps or after certain interval of steps. If the *save\_config* parameter is not specified, the TornasoleHook will store tensors after every 100 steps.

For additional details on TornasoleHook, SaveConfig and Collection please refer to the [API documentation](api.md)

### Register Tornasole hook to the model before starting of the training.

Users can use the hook as a callback function when training a booster.

```
xgboost.train(params, dtrain, callbacks=[hook])
```

 Examples

## API

Please refer to [this document](api.md) for description of all the functions and parameters that our APIs support.

####  Hook

TornasoleHook is the entry point for Tornasole into your program.
Some key parameters to consider when creating the TornasoleHook are the following:

- `out_dir`: This represents the path to which the outputs of tornasole will be written to under a directory with the name `out_dir`. Note that in a SageMaker environment the out_dir will be ignored and always default to `/opt/ml/output/tensors`.
- `save_config`: This is an object of [SaveConfig](#saveconfig). The SaveConfig allows user to specify when the tensors are to be stored. User can choose to specify the number of steps or the intervals of steps when the tensors will be stored. If not specified, it defaults to a SaveConfig which saves every 100 steps.
- `include_collections`: This represents the [collections](#collection) to be saved. With this parameter, user can control which tensors are to be saved.
- `include_regex`: This represents the regex patterns of names of tensors to save. With this parameter, user can control which tensors are to be saved.

**Examples**

- Save evaluation metrics and feature importances every 10 steps to an S3 location:

```
import tornasole.xgboost as tx
tx.TornasoleHook(save_config=SaveConfig(save_interval=10),
                 include_collections=['metrics', 'feature_importance'])
```

- Save custom tensors by regex pattern to a local path

```
import tornasole.xgboost as tx
tx.TornasoleHook(include_regex=['validation*'])
```

Refer [API](api.md) for all parameters available and detailed descriptions.

#### Collection

Collection object helps group tensors for easier handling of tensors being saved.
A collection has its own list of tensors, include regex patterns, and [save config](#saveconfig).
This allows setting of different save configs for different tensors.
These collections are then also available during analysis.
Tornasole will save the value of tensors in collection, if the collection is included in `include_collections` param of the [hook](#hook).

Refer to [API](api.md) for all methods available when using collections such
as setting SaveConfig for a specific collection or retrieving all collections.

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

Refer to [API](api.md) for all parameters available and detailed descriptions for them, as well as example SaveConfig objects.

#### ReductionConfig

ReductionConfig is not currently used in XGBoost Tornasole.
When Tornasole is used with deep learning frameworks, such as MXNet,
Tensorflow, or PyTorch, ReductionConfig allows the saving of certain
reductions of tensors instead of saving the full tensor.
By reduction here we mean an operation that converts the tensor to a scalar.
However, in XGBoost, we currently support evaluation metrics, feature
importances, and average SHAP values, which are all scalars and not tensors.
Therefore, if the `reduction_config` parameter is set in
`tornasole.xgboost.TornasoleHook`, it will be ignored and not used at all.

### How to save tensors

There are different ways to save tensors when using Tornasole.
Tornasole provides easy ways to save certain standard tensors by way of default
collections (a Collection represents a group of tensors).
Examples of such collections are 'metrics', 'feature\_importance',
'average\_shap', and 'default'.
Besides the tensors in above default collections, you can save tensors by name or regex patterns on those names.
This section will take you through these ways in more detail.

#### Saving the tensors with *include\_regex*
The TornasoleHook API supports *include\_regex* parameter. The users can specify a regex pattern with this pattern. The TornasoleHook will store the tensors that match with the specified regex pattern. With this approach, users can store the tensors without explicitly creating a Collection object. The specified regex pattern will be associated with 'default' Collection and the SaveConfig object that is associated with the 'default' collection.

#### Default Collections
Currently, the XGBoost TornasoleHook creates Collection objects for
'metrics', 'feature\_importance', 'average\_shap', and 'default'. These
collections contain the regex pattern that match with
evaluation metrics, feature importances, and SHAP values. The regex pattern for
the 'default' collection is set when user specifies *include\_regex* with
TornasoleHook or sets the *save_all=True*.  These collections use the SaveConfig
parameter provided with the TornasoleHook initialization. The TornasoleHook
will store the related tensors, if user does not specify any special collection
with *include\_collections* parameter. If user specifies a collection with
*include\_collections* the above default collections will not be in effect.
Please refer to [this document](api.md) for description of all the default=
collections.

#### Custom Collections

You can also create any other customized collection yourself.
You can create new collections as well as modify existing collections

##### Creating a collection

Each collection should have a unique name (which is a string). You can create
collections by invoking helper methods as described in the [API](api.md) documentation

```
from tornasole.xgboost as get_collection
get_collection('metrics').include(['validation-auc'])
```

##### Adding tensors

Tensors can be added to a collection by either passing an include regex parameter to the collection.
If you don't know the name of the tensors you want to add, you can also add the tensors to the collection
by the variables representing the tensors in code. The following sections describe these two scenarios.

##### Adding tensors by regex
If you know the name of the tensors you want to save and can write regex
patterns to match those tensornames, you can pass the regex patterns to the collection.
The tensors which match these patterns are included and added to the collection.

```
from tornasole.xgboost import get_collection
get_collection('metrics').include(["train*", "*-auc"])
```

#### Saving All Tensors
Tornasole makes it easy to save all the tensors in the model. You just need to set the flag `save_all=True` when creating the hook. This creates a collection named 'all' and saves all the tensors under that collection.
**NOTE : Storing all the tensors will slow down the training and will increase the storage consumption.**


## ContactUs
We would like to hear from you. If you have any question or feedback, please reach out to us tornasole-users@amazon.com

## License
This library is licensed under the Apache 2.0 License.
