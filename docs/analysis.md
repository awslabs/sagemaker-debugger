# Programming model for analysis

This page describes the programming model that Amazon SageMaker Debugger provides for your analysis, and introduces you to the constructs of trial, tensor and rule.

## Table of Contents
* [Trial](#Trial)
	* [Path of trial](#Path-of-trial)
		* [SageMaker training job](#SageMaker-training-job)
		* [Non-SageMaker training jobs](#Non-SageMaker-training-jobs)
	* [Creating a trial object](#Creating-a-trial-object)
		* [Creating an S3 trial](#Creating-S3-trial)
		* [Creating a local trial](#Creating-local-trial)
		* [Restricting analysis to a range of steps](#Restricting-analysis-to-a-range-of-steps)
	* [Trial API](#Trial-API)
		* [tensor_names](#tensor_names)
		* [tensor](#tensor)
		* [has_tensor](#has_tensor)
		* [steps](#steps)
		* [modes](#modes)
		* [mode](#mode)
		* [mode_step](#mode_step)
		* [global_step](#global_step)
		* [workers](#workers)
		* [collections](#collections)
		* [collection](#collection)
		* [wait\_for\_steps](#wait\_for\_steps)
		* [has\_passed\_step](#has\_passed\_step)
* [Tensor](#Tensor-1)
	* [Tensor API](#Tensor-API)
		* [steps](#steps-1)
		* [value](#value)
		* [reduction_value](#reduction_value)
		* [shape](#shape)
		* [values](#values)
		* [reduction_values](#reduction_values)
		* [shapes](#shapes)
		* [workers](#workers-1)
		* [prev_steps](#prev_steps)
* [Rules](#Rules)
	* [Built-in rules](#Built-In-Rules)
	* [Writing a custom rule](#Writing-a-custom-rule)
		* [Constructor](#Constructor)
		* [Function to invoke at a given step](#Function-to-invoke-at-a-given-step)
	* [Invoking a rule](#Invoking-a-rule)
		* [invoke_rule](#invoke_rule)
* [Exceptions](#Exceptions)
* [Utils](#Utils)
	* [Enable or disable refresh of tensors in a trial](#Enable-or-disable-refresh-of-tensors-in-a-trial)

## Trial
A trial is an object which lets you query for tensors for a given training job, specified by the path where `smdebug`'s artifacts are saved.
A trial is capable of loading new tensors as and when they become available at the given path, allowing you to do both offline as well as realtime analysis.

### Path of trial
#### SageMaker training jobs
When running a SageMaker job, this path is on Amazon S3. SageMaker saves data from your training job locally on the training instance first and uploads them to an Amazon S3 location in your account. When you start a SageMaker training job with the Python SDK, you can control this path using the parameter `s3_output_path` in the `DebuggerHookConfig` object. This is an optional parameter; if you do not pass this, the Python SDK populates a default location for you. If you do pass this, make sure the bucket is in the same Region as the training job. If you're not using the Python SDK, set this path for the parameter `S3OutputPath` in the `DebugHookConfig` section of the `CreateTrainingJob` API. SageMaker takes this path and appends `training_job_name` and `debug-output` to it to ensure a unique path for each training job.

#### Non-SageMaker training jobs
If you are not running a SageMaker training job, this is the path you pass as `out_dir` when you create a `smdebug` [`hook`](api.md#hook). You can pass either a local path or an Amazon S3 path (as `s3://bucket/prefix`).

### Creating a trial object
There are two types of trials you can create: `LocalTrial` or `S3Trial`, depending on the path. We provide a wrapper method to create the appropriate trial.

The parameters you have to provide are:
- `path`: The path can be a local path or an Amazon S3 path of the form `s3://bucket/prefix`. You should see directories such as `collections`, `events` and `index` at this path once the training job starts.
- `name`: A name can be any string. The name helps you manage different trials. This is an optional parameter, which defaults to the basename of the path if it is not passed. Please make sure to give it a unique name to prevent confusion.

#### Creating an Amazon S3 trial
```python
from smdebug.trials import create_trial
trial = create_trial(path='s3://smdebug-testing-bucket/outputs/resnet', name='resnet_training_run')
```

#### Creating a local trial
```python
from smdebug.trials import create_trial
trial = create_trial(path='/home/ubuntu/smdebug_outputs/resnet', name='resnet_training_run')
```

#### Restricting analysis to a range of steps
You can optionally pass `range_steps` to restrict your analysis to a certain range of steps.
Note that if you do so, the trial will not load data from other steps.

*Examples*
- `range_steps=(100, None)`: This will load all steps after 100
- `range_steps=(None, 100)`: This will load all steps before 100
- `range_steps=(100, 200)` : This will load steps between 100 and 200
- `range_steps=None`: This will load all steps

```python
from smdebug.trials import create_trial
tr = create_trial(path='s3://smdebug-testing-bucket/outputs/resnet', name='resnet_training',
                  range_steps=(100, 200))
```

### Trial API

Here's a list of methods that the Trial API provides to help you load data for analysis. Please click on the method to see a detailed description of the method and all the parameters it takes. If you are not familiar with `smdebug` constructs, review [this API documentation](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md) before going through this page.

| Method        | Description           |
| ------------- |-------------|
| [trial.tensor_names()](#tensor_names)      | See names of all tensors available |
| [trial.tensor(name)](#tensor)      | Retrieve `smdebug` tensor object |
| [trial.has_tensor(name)](#has_tensor)      | Query for whether tensor was saved |
| [trial.steps()](#steps) | Query steps for which data was saved |
| [trial.modes()](#modes) | Query modes for which data was saved |
| [trial.mode(step)](#mode) | Query the mode for a given global step |
| [trial.global_step(mode, step)](#global_step) | Query global step for a given step and mode |
| [trial.mode_step(step)](#mode_step) | Query the mode step for a given global step |
| [trial.workers()](#workers) | Query list of workers from the data saved |
| [trial.collections()](#collections) | Query list of collections saved from the training job |
| [trial.collection(name)](#collection) | Retrieve a single collection saved from the training job |
| [trial.wait\_for\_steps(steps)](#wait\_for\_steps) | Wait until the requested steps are available |
| [trial.has\_passed\_step(step)](#has\_passed\_step) | Query whether the requested step is available |


#### tensor_names
Retrieves names of tensors saved.
```python
trial.tensor_names(step= None,
              mode=modes.GLOBAL,
              regex=None,
              collection=None)
```

##### Arguments
All arguments to this method are optional. You are not required to pass any of these arguments as keyword arguments.

- `step (int)`: If you want to retrieve the list of tensors saved at a particular step, pass the step number as an integer. This step number is treated as the step number corresponding to the mode passed as follows. By default, it is treated as global step.
- `mode (smdebug.modes enum value)`: If you want to retrieve the list of tensors saved for a particular mode, pass the mode here as `smd.modes.TRAIN`, `smd.modes.EVAL`, `smd.modes.PREDICT`, or `smd.modes.GLOBAL`.
- `regex (str or list[str])`: You can filter tensors matching regex expressions by passing a regex expression as a string or list of strings. You can only pass one of `regex` or `collection` parameters.
- `collection (Collection or str)`: You can filter tensors belonging to a collection by either passing a collection object or the name of collection as a string. You can only pass one of `regex` or `collection` parameters.

##### Returns
`list[str]`: List of strings representing names of tensors matching the given arguments. Arguments are processed as follows: get the list of tensor names for a given step and mode, and save for a given step matching all the given arguments, i.e. intersection of tensors matching each of the parameters.

##### Examples
- `trial.tensor_names()`: Returns all tensors saved for any step or mode.
- `trial.tensor_names(step=10, mode=modes.TRAIN)`: Returns tensors saved for training step 10.
- `trial.tensor_names(regex='relu')`: Returns all tensors matching the regex pattern `relu` saved for any step or mode.
- `trial.tensor_names(collection='gradients')`: Returns tensors from collection `gradients`.
- `trial.tensor_names(step=10, mode=modes.TRAIN, regex='softmax')`: Returns tensors saved for the 10th training step which match the regex `softmax`.


#### tensor
Retrieve the `smdebug.core.tensor.Tensor` object by the given name `tname`. You can review all the methods that this tensor object provides [here](#Tensor-1).
```python
trial.tensor(tname)
```
##### Arguments
- `tname (str)`: Takes the name of the tensor.

##### Returns
`smdebug.core.tensor.Tensor` object which has [this API](#Tensor-1).

#### has_tensor
Query whether the trial has a tensor by the given name.
```python
trial.has_tensor(tname)
```

##### Arguments
- `tname (str)`: Takes the name of the tensor.

##### Returns
`bool`: `True` if the tensor is seen by the trial so far, else `False`.

#### steps
Retrieve a list of steps seen by the trial.
```python
trial.steps(mode=None)
```

##### Arguments
- `mode (smdebug.modes enum value)`: By passing a mode here, you can retrieve the list of steps seen by a trial for that mode.
If you do not pass this, it returns steps for all modes.

##### Returns
`list[int]` List of integers representing step numbers. If a mode was passed, this returns steps within that mode, called mode steps.
Each of these mode steps has a global step number associated with it. The global step represents
the sequence of steps across all modes executed by the job.

#### modes
Retrieve a list of modes seen by the trial.
```python
trial.modes()
```

##### Returns
`list[smdebug.modes enum value]`: List of modes for which data was saved from the training job across all steps seen.

#### mode
Given a global step number, you can identify the mode for that step using this method.
```python
trial.mode(global_step=100)
```

##### Arguments
- `global_step (int)` Takes the global step as an integer.

##### Returns
`smdebug.modes enum value` of the given global step.

#### mode_step
Given a global step number, you can identify the `mode_step` for that step using this method.
```python
trial.mode_step(global_step=100)
```

##### Arguments
- `global_step (int)`: Takes the global step as an integer.

##### Returns
`int`: An integer representing `mode_step` of the given global step. Typically used in conjunction with the `mode` method.

#### global_step
Given a mode and a `mode_step` number, you can retrieve its global step using this method.
```python
trial.global_step(mode=modes.GLOBAL, mode_step=100)
```

##### Arguments
- `mode (smdebug.modes enum value)`: Takes the mode as an `enum` value.
- `mode_step (int)`: Takes the mode step as an integer.

##### Returns
`int`: An integer representing the `global_step` of the given mode and `mode_step`.

#### workers
Query for all the worker processes from which data was saved by `smdebug` during multi-worker training.
```python
trial.workers()
```

##### Returns
`list[str]`: A sorted list of names of worker processes from which data was saved. If you are using TensorFlow Mirrored Strategy for multi-worker training, these represent names of different devices in the process. For Horovod, torch.distributed, and similar distributed training approaches, these represent names of the form `worker_0`, where 0 is the rank of the process.


#### collections

List the collections from the trial. Note that the tensors part of these collections may not necessarily have been saved from the training job. Whether a collection was saved or not depends on the configuration of the hook during training.

```python
trial.collections()
```

##### Returns
`dict[str -> Collection]`: A dictionary indexed by the name of the collection, with the collection object as the value. See [Collection API](api.md#Collection) for more details.

#### collection

Get a specific collection from the trial. Note that tensors which are part of this collection may not necessarily have been saved from the training job. Whether this collection was saved or not depends on the configuration of the hook during training.

```python
trial.collection(coll_name)
```
##### Arguments
- `coll_name (str)`: Name of the collection.

##### Returns
`Collection`: The requested collection object. See [Collection API](api.md#Collection) for more details.


#### wait\_for\_steps
This method allows you to wait for steps before proceeding. Use this method if you want to wait for `smdebug` to see the required steps so you can then query and analyze the tensors saved by that step. This method blocks until `smdebug` sees all data from the steps.
```python
trial.wait_for_steps(required_steps, mode=modes.GLOBAL)
```

##### Arguments
- `required_steps (list[int])`: Step numbers for which to wait.
- `mode (smdebug.modes enum value)`: The mode to which given step numbers correspond. This defaults to modes.GLOBAL.

##### Returns
None, but it only returns after we know whether we have seen the steps.

##### Exceptions raised
`StepUnavailable` and `NoMoreData`. See the [Exceptions](#exceptions) section for more details.

#### has\_passed\_step
```python
trial.has_passed_step(step, mode=modes.GLOBAL)
```

##### Arguments
- `step (int)`: The step number to check if the trial has passed.
- `mode (smdebug.modes enum value)`: The mode to which a given step number corresponds. This defaults to modes.GLOBAL.

##### Returns
`smdebug.core.tensor.StepState enum value`, which can take one of three values: `UNAVAILABLE`, `AVAILABLE`, and `NOT_YET_AVAILABLE`.

TODO@Nihal describe these in detail

## Tensor
An `smdebug` tensor object can be retrieved through the `trial.tensor(name)` API. It is uniquely identified by the string representing its name.
 It provides the following methods.

| Method | Description|
| ---- | ----- |
| [steps()](#steps-1) | Query steps for which tensor was saved
.|
| [value(step)](#value) | Get the value of the tensor at a given step as a `numpy` array. |
| [reduction_value(step)](#reduction_value) | Get the reduction value of the chosen tensor at a particular step. |
| [reduction_values(step)](#reduction_values) | Get all reduction values saved for the chosen tensor at a particular step. |
| [values(mode)](#values) | Get the values of the tensor for all steps of a given mode. |
| [workers(step)](#workers-1) | Get all the workers for which this tensor was saved at a given step. |
| [prev\_steps(step, n)](#prev_steps) | Get the last `n` step numbers of a given mode from a given step. |

### Tensor API
#### steps
Query for the steps at which the given tensor was saved.
```python
trial.tensor(name).steps(mode=ModeKeys.GLOBAL, show_incomplete_steps=False)
```

##### Arguments
- `mode (smdebug.modes enum value)`: The mode whose steps to return for the given tensor. Defaults to `modes.GLOBAL`.
- `show_incomplete_steps (bool)`: This parameter is relevant only for distributed training. By default, this method only returns the steps which have been received from all workers. But if this parameter is set to `True`, this method returns steps received from at least one worker.

##### Returns
`list[int]`: A list of steps at which the given tensor was saved.

#### value
Get the value of the tensor at a given step as a `numpy` array.
```python
trial.tensor(name).value(step_num, mode=ModeKeys.GLOBAL, worker=None)
```

##### Arguments
- `step_num (int)`: The step number whose value is to be returned for the mode passed through the next parameter.
- `mode (smdebug.modes enum value)`: The mode applicable for the preceding step number passed. Defaults to `modes.GLOBAL`.
- `worker (str)`: This parameter is only applicable for distributed training. You can retrieve the value of the tensor from a specific worker by passing the worker name. You can query all the workers seen by the trial with the `trial.workers()` method. You can also query the workers which saved a value for the tensor at a specific step with the method: `trial.tensor(name).workers(step, mode)`.

##### Returns
`numpy.ndarray`: The value of the tensor at the given step and worker (if the training job saved data from multiple workers).

#### reduction_value
Get the reduction value of the chosen tensor at a particular step. A reduction value is a tensor reduced to a single value through reduction or aggregation operations. The different reductions for which you can query are the same as those allowed in [ReductionConfig](api.md#reductionconfig) when saving tensors.
This API thus allows you to access the reduction you might have saved instead of the full tensor. If you saved the full tensor, it calculates the requested reduction at the time of this call.

Reduction names allowed are `min`, `max`, `mean`, `prod`, `std`, `sum`, `variance`, and `l1`, `l2` representing the norms.

Each of these can be retrieved for the absolute value of the tensor or the original tensor. The preceding example gets the mean of the absolute value of the tensor. `abs` can be set to `False` if you want to see the `mean` of the actual tensor.

If you saved the tensor without any reduction, then you can retrieve the actual tensor as a `numpy` array and compute any reduction in which you might be interested. In such a case, you do not need this method.

```python
trial.tensor(name).reduction_value(step_num, reduction_name,
                                    mode=modes.GLOBAL, worker=None, abs=False)

```
##### Arguments
- `step_num (int)`: The step number whose value is to be returned for the mode passed through the next parameter.
- `reduction_name (str)`: The name of the reduction for which to query. This can be one of `min`, `max`, `mean`, `std`, `variance`, `sum`, `prod`, and the norms `l1`, `l2`.
- `mode (smdebug.modes enum value)`: The mode applicable for the preceding step number passed. Defaults to `modes.GLOBAL`.
- `worker (str)`: This parameter is only applicable for distributed training. You can retrieve the value of the tensor from a specific worker by passing the worker name. You can query all the workers seen by the trial with the `trial.workers()` method. You might also be interested in querying the workers which saved a value for the tensor at a specific step. This is possible with the method: `trial.tensor(name).workers(step, mode)`.
- `abs (bool)`: If `abs` is` True`, this method tries to return the reduction passed through `reduction_name` after taking the absolute value of the tensor. It defaults to `False`.

##### Returns
`numpy.ndarray`: The reduction value of the tensor at the given step and worker (if the training job saved data from multiple workers) as a 1x1 `numpy` array. If this reduction was saved for the tensor during training as part of specification through reduction config, it is loaded and returned. If the given reduction was not saved then, but the full tensor was saved, the reduction is computed on the fly and returned. If both the chosen reduction and full tensor are not available, this method raises a `TensorUnavailableForStep` exception.

#### shape
Get the shape of the chosen tensor at a particular step.

```python
trial.tensor(name).shape(step_num, mode=modes.GLOBAL, worker=None)

```
##### Arguments
- `step_num (int)`: The step number whose value is to be returned for the mode passed through the next parameter.
- `mode (smdebug.modes enum value)`: The mode applicable for the step number passed above. Defaults to `modes.GLOBAL`.
- `worker (str)`: This parameter is only applicable for distributed training. You can retrieve the value of the tensor from a specific worker by passing the worker name. You can query all the workers seen by the trial with the `trial.workers()` method. You can also query the workers which saved a value for the tensor at a specific step with the method: `trial.tensor(name).workers(step, mode)`.

##### Returns
`tuple(int)`:  If only the shape of this tensor was saved through `save_shape` configuration in `ReductionConfig`, it is returned. If the full tensor was saved, then the shape is computed and returned today. If both the shape and full tensor are not available, this method raises a `TensorUnavailableForStep` exception.

#### values
Get the values of the tensor for all steps of a given mode.

```python
trial.tensor(name).values(mode=modes.GLOBAL, worker=None)
```

##### Arguments
- `mode (smdebug.modes enum value)`: The mode applicable for the preceding step number passed. Defaults to `modes.GLOBAL`.
- `worker (str)`: This parameter is only applicable for distributed training. You can retrieve the value of the tensor from a specific worker by passing the worker name. You can query all the workers seen by the trial with the `trial.workers()` method. You can also query the workers which saved a value for the tensor at a specific step. This is possible with the method: `trial.tensor(name).workers(step, mode)`.

##### Returns
`dict[int -> numpy.ndarray]`: A dictionary with step numbers as keys and `numpy` arrays representing the value of the tensor as values.

#### reduction_values
Get all reduction values saved for the chosen tensor at a particular step. A reduction value is a tensor reduced to a single value through reduction or aggregation operations. See the description of the method `reduction_value` for more details.

```python
trial.tensor(name).reduction_values(step_num, mode=modes.GLOBAL, worker=None)
```

##### Arguments
- `step_num (int)`: The step number whose value is to be returned for the mode passed through the next parameter.
- `mode (smdebug.modes enum value)`: The mode applicable for the preceding step number passed. Defaults to `modes.GLOBAL`.
- `worker (str)`: This parameter is only applicable for distributed training. You can retrieve the value of the tensor from a specific worker by passing the worker name. You can query all the workers seen by the trial with the `trial.workers()` method. You can also query the workers which saved a value for the tensor at a specific step. This is possible with the method: `trial.tensor(name).workers(step, mode)`.

##### Returns
`dict[(str, bool) -> numpy.ndarray]`: A dictionary with keys being tuples of the form `(reduction_name, abs)` to a 1x1 `numpy ndarray` value. `abs` here is a boolean that denotes whether the reduction was performed on the absolute value of the tensor or not. Note that this method only returns the reductions which were saved from the training job. It does not compute all known reductions and return them if only the raw tensor was saved.

#### shapes
Get the shapes of the tensor for all steps of a given mode.

```python
trial.tensor(name).shapes(mode=modes.GLOBAL, worker=None)
```

##### Arguments
- `mode (smdebug.modes enum value)`: The mode applicable for the preceding step number passed. Defaults to `modes.GLOBAL`.
- `worker (str)`: This parameter is only applicable for distributed training. You can retrieve the value of the tensor from a specific worker by passing the worker name. You can query all the workers seen by the trial with the `trial.workers()` method. You can also query the workers which saved a value for the tensor at a specific step. This is possible with the method: `trial.tensor(name).workers(step, mode)`.

##### Returns
`dict[int -> tuple(int)]`: A dictionary with step numbers as keys and tuples of ints representing the shapes of the tensor as values.

#### workers
Get all the workers for which this tensor was saved at a given step.

```python
trial.tensor(name).workers(step_num, mode=modes.GLOBAL)
```

##### Arguments
- `step_num (int)`: The step number whose value is to be returned for the mode passed through the next parameter.
- `mode (smdebug.modes enum value)`: The mode applicable for the preceding step number passed. Defaults to `modes.GLOBAL`.

##### Returns
`list[str]`: A list of worker names for which the tensor was saved at the given step.

#### prev_steps
Get the last `n` step numbers of a given mode from a given step.

```python
trial.tensor(name).prev_steps(step, n, mode=modes.GLOBAL)
```
##### Arguments
- `step (int)`: The step number whose value is to be returned for the mode passed.
- `n (int)`: Number of previous steps to return
- `mode (smdebug.modes enum value)`: The mode applicable for the preceding step number passed. Defaults to `modes.GLOBAL`.

##### Returns
`list[int]`: A list of a size at most `n` representing the previous steps for the given step and mode. Note that this list can be of a size less than `n` if there were only fewer than `n` steps saved before the given step in this trial.

## Rules
Rules are the medium by which SageMaker Debugger executes a certain piece of code regularly on different steps of a training job. A rule is assigned to a trial and can be invoked at each new step of the trial. It can also access other trials for its evaluation. You can evaluate a rule using tensors from the current step or any step before the current step. Please ensure your logic respects these semantics, else you get a `TensorUnavailableForStep` exception as the data is not yet available for future steps.

### Built-in rules
Please refer to the built-in rules that SageMaker provides [here](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md#built-in-rules).

### Writing a custom rule
Writing a rule involves implementing the [Rule interface](../smdebug/rules/rule.py). In a following section, we look at a simplified version of a `VanishingGradient` rule.

#### Constructor
Creating a rule involves first inheriting from the base `Rule` class provided by `smdebug`.
For this example rule, we do not need to look at any other trials, so we set `other_trials` to `None`.

```python
from smdebug.rules import Rule

class VanishingGradientRule(Rule):
    def __init__(self, base_trial, threshold=0.0000001):
        super().__init__(base_trial, other_trials=None)
        self.threshold = float(threshold)
```

Apart from `base_trial` and `other_trials` (if required), we require all
arguments of the rule constructor to take a string as a value. You can parse them to the type
that you want from the string. This means that if you want to pass a list of strings, you might want to pass them as a comma-separated string. This restriction is enforced so you can create and invoke rules from JSON using Sagemaker's APIs.

##### Function to invoke at a given step
In this function, you can implement the core logic of what you want to do with these tensors.
It should return a boolean value `True` or `False`, where `True` means the rule evaluation condition has been met. When you invoke these rules through SageMaker, the rule evaluation ends when the rule evaluation condition is met. SageMaker creates an Amazon Cloudwatch event for every rule evaluation job, which can be used to define actions that you might want to take based on the state of the rule.

A simplified version of the actual invoke function for `VanishingGradientRule` is as follows:

```python
    def invoke_at_step(self, step):
        for tensorname in self.base_trial.tensors(collection='gradients'):
            tensor = self.base_trial.tensor(tensorname)
            abs_mean = tensor.reduction_value(step, 'mean', abs=True)
            if abs_mean < self.threshold:
                return True
            else:
                return False
```

That's it! Writing a rule is as simple as that.

### Invoking a rule through SageMaker
After you've written your rule, you can ask SageMaker to evaluate the rule against your training job by using the SageMaker Python SDK:
```
estimator = Estimator(
    ...
    rules = Rules.custom(
    	name='VGRule',
        image_uri='864354269164.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rule-evaluator:latest',
    	instance_type='ml.t3.medium', # instance type to run the rule evaluation on
    	source='rules/vanishing_gradient_rule.py', # path to the rule source file
    	rule_to_invoke='VanishingGradientRule', # name of the class to invoke in the rule source file
    	volume_size_in_gb=30, # EBS volume size required to be attached to the rule evaluation instance
    	collections_to_save=[CollectionConfig("gradients")], # collections to be analyzed by the rule
    	rule_parameters={
      		"threshold": "20.0" # this will be used to initialize 'threshold' param in your rule constructor
    	}
)
```
If you're using the SageMaker API directly to evaluate the rule, then you can specify the rule configuration [`DebugRuleConfigurations`](https://docs.aws.amazon.com/sagemaker/latest/dg/API_DebugRuleConfiguration.html) in the `CreateTrainingJob` API request:
```
"DebugRuleConfigurations": [
	{
		"RuleConfigurationName": "VGRule",
		"InstanceType": "ml.t3.medium",
		"VolumeSizeInGB": 30,
		"RuleEvaluatorImage": "864354269164.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rule-evaluator:latest",
		"RuleParameters": {
			"source_s3_uri": "s3:/path/to/vanishing_gradient_rule.py",
			"rule_to_invoke": "VanishingGradient",
			"threshold": "20.0"
		}
	}
]
```

#### Invoking a rule outside of SageMaker through `invoke_rule`
You can invoke the rule locally during development. We provide a function to invoke rules easily. See [smdebug/rules/rule_invoker.py](../smdebug/rules/rule_invoker.py). The invoke function takes an instance of a rule and invokes it for a series of steps one after the other, and has the following syntax:

```python
from smdebug.rules import invoke_rule
from smdebug.trials import create_trial

trial = create_trial('s3://smdebug-dev-test/mnist-job/')
rule_obj = VanishingGradientRule(trial, threshold=0.0001)
invoke_rule(rule_obj, start_step=0, end_step=None)
```

##### Arguments
- `rule_obj (Rule)`: An instance of a subclass of `smdebug.rules.Rule` that you want to invoke.
- `start_step (int)`: A global step number from which to start invoking the rule. Note that this refers to a global step. This defaults to 0.
- `end_step (int or  None)`: Global step number before which to end the invocation of rule. To clarify, `end_step` is an exclusive bound. The rule is invoked at `end_step`. This defaults to `None`, which means it runs until the end of the job.
- `raise_eval_cond (bool)`: This parameter controls whether to raise the exception `RuleEvaluationConditionMet` when raised by the rule, or to catch it and log the message and move to the next step. It defaults to `False`, which implies that it catches the exception, logs that the evaluation condition was met for a step, and moves on to evaluate the next step.


## Exceptions
`smdebug` is designed to be aware that the tensors required to evaluate a rule may not be available at every step. Hence, it raises a few exceptions which allow us to control what happens when a tensor is missing. These are available in the `smdebug.exceptions` module. You can import them as follows:

```python
from smdebug.exceptions import *
```

Here are the exceptions (along with others) and their meaning:

- `TensorUnavailableForStep`: This means that the tensor requested is not available for the step. It may have been or will be saved for a different step number. You can check which steps the tensor is saved for by `trial.tensor('tname').steps()` [api](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#steps-1). Note that this exception implies that the requested tensor will never become available for this step in the future.

- `TensorUnavailable`: This means that this tensor has not been saved from the training job. Note that if you have a `SaveConfig` which saves a certain tensor only after the time you queried for the tensor, you might get a `TensorUnavailable` exception even if the tensor may become available later for some step.

- `StepUnavailable`: This means that the step was not saved from the training job. No tensor will be available for this step.

- `StepNotYetAvailable`: This means that the step has not yet been seen from the training job. It may be available in the future if the training is still going on. We automatically load new data as and when it becomes available. This step may either become available in the future, or the exception might change to `StepUnavailable`.

- `NoMoreData`: This is raised when the training ends. Once you see this, you know that there are no more steps and no more tensors saved.

- `RuleEvaluationConditionMet`: This is raised when the rule invocation returns `True` for some step.

- `MissingCollectionFiles`: This is raised when no data was saved by the training job. Check that the `hook` was configured correctly before starting the training job.

## Utils

### Enable or disable refresh of tensors in a trial

By default, `smdebug` refreshes tensors each time you try to query the tensor.
It looks for whether this tensor is saved for new steps and, if so, fetches them.
If you know the saved data will not change (stopped the machine learning job), or
are not interested in the latest data, you can stop the refreshing of tensors as follows:

`no_refresh` takes a trial or a list of trials, which should not be refreshed.
Anything executed inside with a `no_refresh` block is not refreshed.

```python
from smdebug.analysis.utils import no_refresh
with no_refresh(trials):
    pass
```

Similarly, if you want to refresh tensors only within a block, you can do so as follows:

```python
from smdebug.analysis.utils import refresh
with refresh(trials):
    pass
```

During rule invocation, `smdebug` waits until the current step is available and then turns off refresh to ensure that you do not get different results for methods like `trial.tensor(name).steps()` and run into subtle issues.
