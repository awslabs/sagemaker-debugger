# Tornasole for XGBoost

Tornasole is a new capability of Amazon SageMaker that allows debugging machine learning training. Tornasole helps you to monitor your training in near real time using rules and would provide you alerts, once it has detected inconsistency in training.

Using Tornasole is a two step process:

**Saving tensors**
This needs the `tornasole` package built for the appropriate framework. This package lets you collect the tensors you want at the frequency that you want, and save them for analysis. 
Please follow the appropriate Readme page to install the correct version. This page is for using Tornasole with XGBoost.

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

#### Instructions

**Make sure that your aws account is whitelisted for Tornasole. [ContactUs](#contactus)**.

Once your account is whitelisted, you should be able to install the `tornasole` package built for XGBoost as follows:

```
aws s3 sync s3://tornasole-binaries-use1/tornasole_xgboost/py3/latest/ tornasole_xgboost/
pip install tornasole_xgboost/tornasole-*
```

**Please note** : If, while installing tornasole, you get a version conflict issue between botocore and boto3, 
you might need to run the following
```
pip uninstall -y botocore boto3 aioboto3 aiobotocore && pip install botocore==1.12.91 boto3==1.9.91 aiobotocore==0.10.2 aioboto3==6.4.1   
```

## Quickstart

If you want to quickly run some examples, you can jump to [examples](#examples) section. You can also see this [XGBoost notebook example](../../examples/xgboost/notebooks/xgboost_abalone.ipynb) to see tornasole working.

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
    output_s3_uri = 's3://my_xgboost_training_debug_bucket/12345678-abcd-1234-abcd-1234567890ab'
    hook = TornasoleHook(out_dir=output_s3_uri, save_config=save_config)
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

### Tornasole local mode example 

The example [xgboost\_abalone\_basic\_hook\_demo.py](../../examples/xgboost/scripts/xgboost_abalone_basic_hook_demo.py) is implemented to show how Tornasole is useful in detecting when the evaluation metrics such as validation error stops decreasing.

```
python3 examples/xgboost/scripts/xgboost_abalone_basic_hook_demo.py --tornasole_path ~/tornasole-testing/basic-demo/trial-one
```

You can monitor the job by using [rules](../rules/README.md). For example, you
can monitor if the metrics such as `train-rmse` or `validation-rmse` in the
`metric` collection stopped decreasing by doing the following:

```
python3 -m tornasole.rules.rule_invoker --trial-dir ~/tornasole-testing/basic-demo/trial-one --rule-name LossNotDecreasing --use_loss_collection False --collection_names 'metric'
``` 
 
Note: You can also try some further analysis on tensors saved by following [programming model](../rules/README.md#the-programming-model) section of our Rules README.

##### Tornasole S3 mode example

```
python3 examples/xgboost/scripts/xgboost_abalone_basic_hook_demo.py --output_uri s3://tornasole-testing/basic-demo/trial-one
```

You can monitor the job for non-decreasing metrics by doing the following:

```
python3 -m tornasole.rules.rule_invoker --trial-dir s3://tornasole-testing/basic-demo/trial-one --rule-name LossNotDecreasing --use_loss_collection False --collection_names 'metric'
``` 
Note: You can also try some further analysis on tensors saved by following [programming model](../rules/README.md#the-programming-model) section of our Rules README.

## API
Please refer to [this document](api.md) for description of all the functions and parameters that our APIs support.

####  Hook

TornasoleHook is the entry point for Tornasole into your program.
Some key parameters to consider when creating the TornasoleHook are the following:

- `out_dir`: This represents the path to which the outputs of tornasole will be written to under a directory with the name `out_dir`. This can be a local path or an S3 prefix of the form `s3://bucket_name/prefix`.
- `save_config`: This is an object of [SaveConfig](#saveconfig). The SaveConfig allows user to specify when the tensors are to be stored. User can choose to specify the number of steps or the intervals of steps when the tensors will be stored. If not specified, it defaults to a SaveConfig which saves every 100 steps.
- `include_collections`: This represents the [collections](#collection) to be saved. With this parameter, user can control which tensors are to be saved.
- `include_regex`: This represents the regex patterns of names of tensors to save. With this parameter, user can control which tensors are to be saved.
 
**Examples**

- Save evaluation metrics and feature importances every 10 steps to an S3 location:

```
import tornasole.xgboost as tx
tx.TornasoleHook(out_dir='s3://tornasole-testing/trial_job_dir', 
                 save_config=SaveConfig(save_interval=10), 
                 include_collections=['metric', 'feature_importance'])
```

- Save custom tensors by regex pattern to a local path

```
import tornasole.xgboost as tx
tx.TornasoleHook(out_dir='/home/ubuntu/tornasole-testing/trial_job_dir',
                 include_regex=['validation*'])
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

- `save_interval`: This allows you to save tensors every `n` steps
- `save_steps`: Allows you to pass a list of step numbers at which tensors should be saved
 
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
Examples of such collections are 'metric', 'feature\_importance',
'average\_shap', and 'default'.
Besides the tensors in above default collections, you can save tensors by name or regex patterns on those names. 
This section will take you through these ways in more detail. 

#### Saving the tensors with *include\_regex*
The TornasoleHook API supports *include\_regex* parameter. The users can specify a regex pattern with this pattern. The TornasoleHook will store the tensors that match with the specified regex pattern. With this approach, users can store the tensors without explicitly creating a Collection object. The specified regex pattern will be associated with 'default' Collection and the SaveConfig object that is associated with the 'default' collection.

#### Default Collections
Currently, the XGBoost TornasoleHook creates Collection objects for
'metric', 'feature\_importance', 'average\_shap', and 'default'. These
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
get_collection('metric').include(['validation-auc'])
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
get_collection('metric').include(["train*", "*-auc"])
```

#### Saving All Tensors
Tornasole makes it easy to save all the tensors in the model. You just need to set the flag `save_all=True` when creating the hook. This creates a collection named 'all' and saves all the tensors under that collection.
**NOTE : Storing all the tensors will slow down the training and will increase the storage consumption.**


### More Examples

| Example Type   | Logging Evluation Metrics |
| -------------- | ------------------------  |
| Link to Example   | [xgboost\_abalone\_basic\_hook\_demo.py](../../examples/xgboost/scripts/xgbost_abalone_basic_hook_demo.py) |

#### Logging evaluation metrics and feature importances of the model

The [xgboost\_abalone\_basic\_hook\_demo.py](../../examples/xgboost/scripts/xgboost_abalone_basic_hook_demo.py) shows end to end example of how to create and register Tornasole hook that can log performance metrics, feature importances, and SHAP values.

Here is how to create a hook for this purpose:

```
# Create a tornasole hook. The initialization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initialization that enables logging of
# evaluation metrics, feature importances, and SHAP values.
def create_tornasole_hook(output_s3_uri, shap_data=None):

    save_config = SaveConfig(save_interval=5)
    hook = TornasoleHook(
        out_dir=output_s3_uri,
        save_config=save_config,
        shap_data=shap_data)

    return hook
```

Here is how to use the hook as a callback function:

```
    bst = xgboost.train(
        params=params, dtrain=dtrain,
        ...
        callbacks=[hook])
```

The example can be invoked as shown below. **Ensure that the s3 bucket specified in command line is accessible for read and write operations**

```
python3 examples/xgboost/scripts/xgboost_abalone_basic_hook_demo.py --output_uri s3://tornasole-testing/basic-xgboost-hook
```

For detail command line help run

```
python3 examples/xgboost/scripts/xgboost_abalone_basic_hook_demo.py --help
```


## Analyzing the Results

This library enables users to collect the desired tensors at desired frequency
while XGBoost training job is running. 
The tensor data generated during this job can be analyzed with various 
rules that check for performance metrics, feature importances, etc.
For example, the performance metrics generated in
[xgboost_abalone.ipynb](../../examples/xgboost/notebooks/xgboost_abalone.ipynb) 
are analyzed by 'LossNotDecreasing' rule, which shows the number of performance
metrics that are not decreasing at regular step intervals.

```
python3 -m tornasole.rules.rule_invoker --trial-dir s3://tornasole-testing/basic-demo/trial-one --rule-name LossNotDecreasing --use_loss_collection False --collection_names 'metric'
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
