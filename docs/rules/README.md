# Tornasole Analysis
Tornasole is an upcoming AWS service designed to be a debugger for machine learning models.
It lets you go beyond just looking at scalars like losses and accuracies during training and gives you
full visibility into all tensors 'flowing through the graph' during training or inference.

Tornasole's analysis module helps you analyze tensors saved from machine learning jobs.
It allows you to run Rules on these tensors as well as anything else you might want to do with
access to raw tensors such as inspection or visualization. It provides access to the tensors in the form of numpy arrays.

## The Programming Model
The library is organized using the following constructs.

### Trial
Trial the construct which lets you query for tensors for a given Tornasole run, specified by the path in which Tornasole artifacts are being saved or were saved.
You can pass a path which holds data for a past run (which has ended) as well as a path for a current run (to which tensors are being written).
Trial is capable of loading new tensors as and when they become available at the given location.

There are two types of trials you can create: LocalTrial or S3Trial.
We provide a wrapper method to create the appropriate trial.

The parameters you have to provide are:
- `name`: name can be any string. It is to help you manage different trials.
Make sure to give it a unique name to prevent confusion.
- `path`: path can be a local path or an S3  path of the form `s3://bucket/prefix`. This path should be where Tornasole hooks (TF or MXNet) save data to.
You should see the directory `events` and the file `collections.json` in this path.

##### Creating local trial
```
from smdebug.trials import create_trial
trial = create_trial(path='/home/ubuntu/tornasole_outputs/train',
                     name='resnet_training_run')
```
##### Creating S3 trial
```
from smdebug.trials import create_trial
trial = create_trial(path='s3://tornasole-testing-bucket/outputs/resnet',
                     name='resnet_training_run')
```
###### Restricting analysis to a range of steps
To any of these methods you can optionally pass `range_steps` to restrict your analysis to a certain range of steps.
Note that if you do so, Trial will not load data from other steps.

*Examples*
- `range_steps=(100, None)`: This will load all steps after 100
- `range_steps=(None, 100)`: This will load all steps before 100
- `range_steps=(100, 200)` : This will load steps between 100 and 200
- `range_steps=None`: This will load all steps

```
lt = create_trial(path='ts_outputs/resnet', name='resnet_training',
                  range_steps=(100, 200))
```


### Mode
A machine learning job can be executing steps in multiple modes, such as training, evaluating, or predicting.
Tornasole provides you the construct of a `mode` to keep data from these modes separate
and make it easy for analysis. To leverage this functionality you have to
call the `set_mode` function of hook such as the following call `hook.set_mode(modes.TRAIN)`.
The different modes available are `modes.TRAIN`, `modes.EVAL` and `modes.PREDICT`.

When you set a mode, steps in that mode have a sequence. We refer to these numbers
as `mode_step`. Each `mode_step` has a global step number associated with it, which represents the
sequence of steps across all modes executed by the job.

For example, your job executes 10 steps, out of which the first 4 are training steps, 5th is evaluation step, 6-9 are training steps, and 10th is evaluation step.
Please note that indexing starts from 0.
In such a case, when you query for the global steps as below:
```
trial.steps()
```
you will see `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.

If you query for training steps as below:
```
from tornasole_rules import modes
trial.steps(modes.TRAIN)
```
 you will see `[0, 1, 2, 3, 4, 5, 6, 7, 8]` because there were 8 training step.
The training step with mode_step 4 here refers to the global step number 5.
You can query this as follows:
```
trial.global_step(mode=modes.TRAIN, mode_step=4)
```

If you did not explicitly set a mode during the running of the job,
the steps are global steps, and are in the `modes.GLOBAL` mode.
In such a case, `global_step` is the same as `mode_step` where mode is `modes.GLOBAL`.

Below, we describe the above functions and others that the Trial API provides.

#### Trial API
Once you have a trial object you can do the following

**See names of all tensors available**

```
trial.tensors()
```
This returns tensors seen for any mode if mode was set during the machine learning job.

**See all steps seen by the Trial for a particular mode**

The returned list is the step number within that mode.
Each of these mode steps has a global step number associated with it.
The global step represents the sequence of steps across all modes executed by the job.

```
from smdebug import modes
trial.steps(mode=modes.TRAIN)
```

**See all global steps seen by the Trial**

This is the list of steps across all modes.

```
trial.steps()
```

**Get the mode and step number within mode for a given global step**

You can get the `mode` of `global_step` 100 as follows:

```
mode = trial.mode(global_step=100)
```

You can get the `mode_step` for `global_step` 100 as follows:

```
mode_step = trial.mode_step(global_step=100)
```

**Know the global step number for a given mode step**

```
from smdebug import modes
global_step_num = trial.global_step(modes.TRAIN, mode_step=10)
```

**See all modes for which the trial has data**

```
trial.modes()
```

**Access a particular tensor**

A tensor is identified by a string which represents its name.

```
trial.tensor('relu_activation:0')
```

**See the global steps for which tensor's value was saved**

```
trial.tensor('relu_activation:0').steps()
```

**See the steps for a given mode when tensor's value was saved**

This returns the mode steps for those steps when this tensor's value was saved for this mode.

```
from smdebug import modes
trial.tensor('relu_activation:0').steps(mode=modes.TRAIN)
```

**Get the value of the tensor at a global step**

This returns the tensor value as a numpy array for the 10th global step.

```
trial.tensor('relu_activation:0').value(10)
```

Please note that this can raise exceptions if the step is not available.
Please see [this section](#when-a-tensor-is-not-available-during-rule-execution) for more details on the different exceptions that can be raised.

**Get the value of the tensor at a step number for a given mode**

This returns the tensor value as a numpy array for the 10th training step.

```
from smdebug import modes
trial.tensor('relu_activation:0').value(10, mode=modes.TRAIN)
```

Please note that this can raise exceptions if the step is not available.
Please see [this section](#when-a-tensor-is-not-available-during-rule-execution) for more details on the different exceptions that can be raised.

**Get reduction value of a tensor at a step**

Tornasole provides a few reductions out of the box that you can query with the following API.
This below returns the mean of the absolute values at step 10.

```
trial.tensor('relu:0').reduction_value(10, 'mean', abs=True)
```

The different reductions you can query for are the same as what are allowed in [ReductionConfig](https://github.com/awslabs/tornasole_tf/blob/master/docs/api.md) when saving tensors.
This API thus allows you to access the reduction you might have saved instead of the full tensor.
If you had saved the full tensor, it will calculate the requested reduction now and cache it.

- `min`, `max`, `mean`, `prod`, `std`, `sum`, `variance`
- `l1`, `l2` norms

Each of these can be retrieved for the absolute value of the tensor or the original tensor.
Above was an example to get the mean of the absolute value of the tensor.
`abs` can be set to `False` if you want to see the `mean` of the actual tensor.

*Note that if you had only saved a particular reduction, you will not be able
to access the full tensor value or any other reduction during analysis.
This also applies to the `abs` flag, meaning that if you had saved the
`mean` of `abs` values of the tensor you can not query for the non absolute values mean.
If you do so, Tornasole will return `None`.*

If you had saved the tensor without any reduction, then you can retrieve the actual tensor
as a numpy array and compute any function you might be interested in.

Please note that this can raise exceptions if the step is not available.
Please see [this section](#when-a-tensor-is-not-available-during-rule-execution) for more details on the different exceptions that can be raised.

**Get names of tensors matching regex**

This method takes a regex pattern or a list of regex patterns.
Each regex pattern is a python style regex pattern string.

```
trail.tensors_matching_regex(['relu_activation*'])
```

**List tensors in a collection**

This returns names of all tensors saved in a given collection.
`gradients` below is the name of the collection we are interested in.

```
trial.tensors_in_collection('gradients')
```

**List collections**

Below returns all collections belonging to the trial as a dictionary.
This dictionary is indexed by the name of the collection, and the value is the collection object.

```
trial.collections()
```

**Refresh or do not refresh tensors**

By default Tornasole refreshes tensors each time you try to query the tensor.
It looks for whether this tensor is saved for new steps and if so fetches them.
If you know the saved data will not change (stopped the machine learning job), or
are not interested in the latest data, you can stop the refreshing of tensors as follows:

`no_refresh` takes a trial or a list of trials, which should not be refreshed.
Anything executed inside the with `no_refresh` block will not be refreshed.

```
from smdebug.analysis.utils import no_refresh
with no_refresh(trials):
    pass
```

Similarly if you want to refresh tensors only within a block, you can do:

```
from smdebug.analysis.utils import refresh
with refresh(trials):
    pass
```

#### When a tensor is not available
Tornasole is designed to be aware that tensors required to execute a rule may not be available at every step.
Hence it raises a few exceptions which allow us to control what happens when a tensor is missing.
These are available in the `smdebug.exceptions` module. You can import them as follows:

```
from smdebug.exceptions import *
```

Here are the exceptions and their meanings:

- `TensorUnavailableForStep` : This means that the tensor requested is not available for the step. This might mean that
this step might not be saved at all by the hook, or that this step might have saved some tensors but the requested
tensor is not part of them. Note that when you see this exception, it means that this tensor can never become available
for this step in the future.

- `TensorUnavailable` : This means that this tensor is not being saved or has not been saved by smdebug. This means
that this tensor will never be seen for any step in smdebug.

- `StepUnavailable`: This means that the step was not saved and Tornasole has no data from the step.

- `StepNotYetAvailable`: This means that the step has not yet been seen by smdebug. It may be available in the future if the training is still going on.
Tornasole automatically loads new data as and when it becomes available.

- `NoMoreData` : This will be raised when the training ends. Once you see this, you will know that there will be no more steps and no more tensors saved.

### Rules
Rules are the medium by which Tornasole executes a certain piece of code regularly on different steps of the jobs.
A rule is assigned to a trial and can be invoked at each new step of the trial.
It can also access other trials for its execution.
You can evaluate a rule using tensors from the current step or any step before the current step.
Please ensure your logic respects these semantics, else you will get a `TensorUnavailableForStep`
exception as the data would not yet be available.

#### Writing a rule
Writing a rule involves implementing the [Rule interface](../../smdebug/rules/rule.py).

##### Constructor
Creating a rule involves first inheriting from the base Rule class Tornasole provides.
For this rule here we do not need to look at any other trials, so we set `other_trials` to None.

```
from smdebug.rules import Rule

class VanishingGradientRule(Rule):
    def __init__(self, base_trial, threshold=0.0000001):
        super().__init__(base_trial, other_trials=None)
        self.threshold = float(threshold)
```

Please note that apart from `base_trial` and `other_trials` (if required), we require all
arguments of the rule constructor to take a string as value. You can parse them to the type
that you want from the string. This means if you want to pass
a list of strings, you might want to pass them as a comma separated string. This restriction is
being enforced so as to let you create and invoke rules from json using Sagemaker's APIs.

##### Function to invoke at a given step
In this function you can implement the core logic of what you want to do with these tensors.

It should return a boolean value `True` or `False`.
This can be used to define actions that you might want to take based on the output of the rule.

A simplified version of the actual invoke function for `VanishingGradientRule` is below:
```
    def invoke_at_step(self, step):
        for tensor in self.base_trial.tensors_in_collection('gradients'):
            abs_mean = tensor.reduction_value(step, 'mean', abs=True)
            if abs_mean < self.threshold:
                return True
            else:
                return False
```

##### Optional: RequiredTensors

This is an optional construct that allows Tornasole to bulk-fetch all tensors that you need to
execute the rule. This helps the rule invocation be more performant so it does not fetch tensor values from S3 one by one. To use this construct, you need to implement a method which lets Tornasole know what tensors you are interested in for invocation at a given step.
This is the `set_required_tensors` method.

Before we look at how to define this method, let us look at the API for `RequiredTensors` class which
needs to be used by this method. An object of this class is provided as a member of the rule class, so you can access it as `self.req_tensors`.

**[RequiredTensors](../../smdebug/rules/req_tensors.py) API**

***Adding a required tensor***
When invoking a rule at a given step, you might require the values of a tensor at certain steps.
This method allows you to specify these steps as required for the tensor.
```
self.req_tensors.add(name=tname,
                     steps=[step_num],
                     trial=None,
                     should_match_regex=False)
```

The arguments are described below:

- `name`: name of the tensor
- `steps`: list of integers representing global step numbers at which this rule requires the values of this tensor
- `trial`: the trial whose tensor values are required. If this argument is None, it is assumed to
take the value of `self.base_trial` in the rule class. None is the default value for this argument.
- `should_match_regex`: boolean which when True means that the given name is treated as a regex pattern.
In such a case, all tensor names in the trial which match that regex pattern are treated as required
for the invocation of the rule at the given step.

***Fetching required tensors***

If required tensors were added inside `set_required_tensors`, during rule invocation it is
automatically used to fetch all tensors at once by calling `req_tensors.fetch()`.
It can raise the exceptions `TensorUnavailable` and `TensorUnavailableForStep` if the trial does not have that tensor, or if the tensor value is not available for the requested step.


If required tensors were added elsewhere, or later, you can call the `req_tensors.fetch()` method
yourself to fetch all tensors at once.

***Querying required tensors***

You can then query the required tensors
*Get names of required tensors*

This method returns the names of the required tensors for a given trial.
```
self.req_tensors.get_names(trial=None)
```
- `trial`: the trial whose required tensors are being queried. If this argument is None, it is assumed to
take the value of `self.base_trial` in the rule class. None is the default value for this argument.

*Get steps for a given required tensor*

This method returns the steps for which the tensor is required to execute the rule at this step.
```
self.req_tensors.get_tensor_steps(name, trial=None)
```
- `trial`: the trial whose required tensors are being queried. If this argument is None, it is assumed to
take the value of `self.base_trial` in the rule class. None is the default value for this argument.


*Get required tensors*

This method returns the list of required tensors for a given trial as `Tensor` objects.
```
self.req_tensors.get(trial=None)
```
- `trial`: the trial whose required tensors are being queried. If this argument is None, it is assumed to
take the value of `self.base_trial` in the rule class. None is the default value for this argument.


###### Declare required tensors
Here, let us define the `set_required_tensors` method to declare the required tensors
to execute the rule at a given `step`.
If we require the gradients of the base_trial to execute the rule at a given step,
then it would look as follows:
```
    def set_required_tensors(self, step):
        for tname in self.base_trial.tensors_in_collection('gradients'):
            self.req_tensors.add(tname, steps=[step])
```

This function will be used by the rule execution engine to fetch all the
required tensors before it executes the rule.
The rule invoker executes the `set_required_tensors` and `invoke_at_step`
methods within a single `no_refresh` block, hence you are guaranteed that the
tensor values or steps numbers will stay the same during multiple calls.

#### Executing a rule
Now that you have written a rule, here's how you can execute it. We provide a function to invoke rules easily.
Refer [smdebug/rules/rule_invoker.py](../../smdebug/rules/rule_invoker.py)
The invoke function has the following syntax.
It takes a instance of a Rule and invokes it for a series of steps one after the other.

```
from smdebug.rules import invoke_rule
invoke_rule(rule_obj, start_step=0, end_step=None)
```

You can invoking the VanishingGradientRule is
```
trial_obj = create_trial(trial_dir)
vr = VanishingGradientRule(base_trial=trial_obj, threshold=0.0000001)
invoke_rule(vr, start_step=0, end_step=1000)
```

For first party Rules (see below) that we provide a rule_invoker module that you can use to run them as follows. You can pass any arguments that the rule takes as command line arguments.

```
python -m smdebug.rules.rule_invoker --trial-dir ~/ts_outputs/vanishing_gradients --rule-name VanishingGradient --threshold 0.0000000001
```

```
python -m smdebug.rules.rule_invoker --trial-dir s3://tornasole-runes/trial0 --rule-name UnchangedTensor --tensor_regex .* --num_steps 10
```

#### First party rules
We provide a few rules which we built. These are supposed to be general purpose rules that you can use easily.
We also hope these serve as examples for you to build your own rules. These are described in [FirstPartyRules](FirstPartyRules.md).


## Examples

We have end-to-end flow example from saving tensors to plotting using saved tensors for [MXNet](../../examples/mxnet/notebooks) and [PyTorch](../../examples/pytorch/notebooks).


## ContactUs
We would like to hear from you. If you have any question or feedback,
please reach out to us tornasole-users@amazon.com

## License
This library is licensed under the Apache 2.0 License.
