# First Party Rules

XGBoost Note: Not all of the following rules are applicable to XGBoost.
For XGBoost, you can use one of `SimilarAcrossRuns`, `AllZero`, `UnchangedTensor`, `LossNotDecreasing`, and `Confusion`.

### VanishingGradient
This rule helps you identify if you are running into a situation where your gradients vanish, i.e. have a
really low or zero magnitude.

Here's how you import and instantiate this rule.
Note that it takes two parameters, `base_trial` the trial whose execution will invoke the rule, and a `threshold` which is
used to determine whether the gradient is `vanishing`. Gradients whose mean of absolute values are lower than this threshold
will return True when we invoke this rule.

The default threshold is `0.0000001`.

```
from tornasole.rules.generic import VanishingGradient
r = VanishingGradient(base_trial, threshold=0.0000001)
```

You can execute this rule in SageMaker by specifying it as follows:
```python
rules_specification = [
    {
        "RuleName": "VanishingGradient",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "threshold" : "0.00000001"
        }
    }
]
```

This rule is not applicable for using with XGBoost.

### ExplodingTensor
This rule helps you identify if you are running into a situation where any tensor has non finite values.

Note that it takes two parameters, `base_trial` the trial whose execution will invoke the rule and
`only_nan` which can be set to True if you only want to monitor for `nan` and not for `infinity`. By default, `only_nan` is set to False, which means it treats `nan` and `infinity` as exploding.

You can execute this rule in SageMaker by specifying it as follows:
```python
rules_specification = [
    {
        "RuleName": "ExplodingTensor",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "only_nan" : "False"
        }
    }
]
```

This rule is not applicable for using with XGBoost.

### SimilarAcrossRuns
This rule helps you compare tensors across runs. Note that this rule takes two trials as inputs. First trial is the `base_trial` whose
execution will invoke the rule, and the `other_trial` is what is used to compare this trial's tensors with.
The third argument is a regex pattern which can be used to restrict this comparision to certain tensors. If this is not passed, it includes all tensors by default.

It returns `True` if tensors are different at a given step between the two trials.

```python
rules_specification = [
    {
        "RuleName": "SimilarAcrossRuns",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "include_regex" : ".*"
        }
    }
]
```

This rule is applicable for using with XGBoost.

### WeightUpdateRatio
This rule helps you keep track of the ratio of the updates to weights during training.  It takes the following arguments:

- `base_trial`: The trial whose execution will invoke the rule. The rule will inspect the tensors gathered during this trial.
- `large_threshold`: float, defaults to 10.0: maximum value that the ratio can take before rule returns True
- `small_threshold`: float, defaults to 0.00000001: minimum value that the ratio can take. the rule returns True if the ratio is lower than this
- `epsilon`: float, defaults to 0.000000001: small constant to ensure that we do not divide by 0 when computing ratio

This rule returns True if the ratio of updates to weights is larger than `large_threshold` or when this ratio is smaller than `small_threshold`.

It is a good sign for training when the updates are in a good scale
compared to the gradients. Very large updates can push weights away from optima,
and very small updates mean slow convergence.

**Note that for this rule to be executed, weights have to be available for two consecutive steps, so save_interval needs to be 1**

```python
rules_specification = [
    {
        "RuleName": "SimilarAcrossRuns",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "include_regex" : ".*"
        }
    }
]
```

This rule is not applicable for using with XGBoost.

### AllZero
This rule helps to identify whether the tensors contain all zeros. It takes following arguments

- `base_trial`: The trial whose execution will invoke the rule. The rule will inspect the tensors gathered during this trial.
- `collection_names`: The list of collection names. The rule will inspect the tensors that belong to these collections.
- `tensor_regex`: The list of regex patterns. The rule will inspect the tensors that match the regex patterns specified in this list.

For this rule, users must specify either the `collection_names` or `tensor_regex` parameter. If both the parameters are specified the rule will inspect union on tensors.

```python
rules_specification = [
    {
        "RuleName": "AllZero",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "tensor_regex" : "input*",
            "collection_names": "weights,bias"
        }
    }
]
```

This rule is applicable for using with XGBoost. To use this rule with XGBoost, specify either the `collection_names` or `tensor_regex` parameter.

### UnchangedTensor
This rule helps to identify whether a tensor is not changing across steps.
This rule runs `numpy.allclose` method to check if the tensor is unchanged.
It takes following arguments

- `base_trial`: The trial whose execution will invoke the rule.
The rule will inspect the tensors gathered during this trial.
- `collection_names`: The list of collection names.
The rule will inspect the tensors that belong to these collections.
If both collection_names and tensor_regex are specified, the rule will check for union of tensors.
- `tensor_regex`: The list of regex patterns.
The rule will inspect the tensors that match the regex patterns specified in this list.
If both collection_names and tensor_regex are specified, the rule will check for union of tensors.
- `num_steps`: int (default is 3). Number of steps across which we check if the tensor has changed.
Note that this checks the last num_steps that are available.
They need not be consecutive.
If num_steps is 2, at step `s` it does not necessarily check for s-1 and s.
If s-1 is not available, it checks the last available step along with s.
In that case it checks the last available step with the current step.
- `rtol`: The relative tolerance parameter, as a float, to be passed to numpy.allclose.
- `atol`: The absolute tolerance parameter, as a float, to be passed to numpy.allclose
- `equal_nan`: Whether to compare NaN’s as equal. If True, NaN’s in a will be considered
equal to NaN’s in b in the output array. This will be passed to numpy.allclose method

For this rule, users must specify either the `collection_names` or `tensor_regex` parameter.
If both the parameters are specified the rule will inspect union on tensors.

```python
rules_specification = [
    {
        "RuleName": "UnchangedTensor",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "tensor_regex" : ".*",
            "num_steps": "5"
        }
    }
]
```

This rule is applicable for using with XGBoost. To use this rule with XGBoost, specify either the `collection_names` or `tensor_regex` parameter.

### LossNotDecreasing
This rule helps you identify if you are running into a situation where loss is not going down.
Note that these losses have to be scalars. It takes the following arguments.

- `base_trial`: The trial whose execution will invoke the rule.
The rule will inspect the tensors gathered during this trial.
- `collection_names`: The list of collection names.
The rule will inspect the tensors that belong to these collections.
Note that only scalar tensors will be picked.
- `tensor_regex`: The list of regex patterns.
The rule will inspect the tensors that match the regex patterns specified in this list.
Note that only scalar tensors will be picked.
- `use_losses_collection`: bool (default is True)
If this is True, it looks for losses from the collection 'losses' if present.
- `num_steps`: int (default is 10). The minimum number of steps after which
we want which we check if the loss has decreased. The rule evaluation happens every num_steps, and
the rule checks the loss for this step with the loss at the newest step
which is at least num_steps behind the current step.
For example, if the loss is being saved every 3 steps, but num_steps is 10. At step 21, loss
for step 21 is compared with the loss for step 9. The next step where loss is checked is at 33,
since 10 steps after 21 is 31, and at 31 and 32 loss is not being saved.
- `diff_percent`: float (default is 0.0)  (between 0.0 and 100.0)
The minimum difference in percentage that loss should be lower by. By default, the rule just checks if loss is going down. If you want to specify a stricter check that loss is going down fast enough, you might want to pass diff_percent.
- `mode`: string
The name of tornasole mode to query tensor values for rule checking. 
If this is not passed, the rule checks for eval mode, then training mode and then global mode in this order.

```python
rules_specification = [
    {
        "RuleName": "LossNotDecreasing",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "use_losses_collection" : "False",
            "tensor_regex": "loss*",
            "min_difference": "20"
        }
    }
]
```

This rule is applicable for using with XGBoost. To use this rule with XGBoost, set `use_loss_collection` to `False` and specify either the `collection_names` or `tensor_regex` parameter.

### Confusion
This rule evaluates the goodness of a confusion matrix for a classification problem.
It creates a matrix of size `category_no` X `category_no` and populates it with data coming
from (`labels`, `predictions`) pairs. For each (`labels`, `predictions`) pairs the count in
`confusion[labels][predictions]` is incremented by 1.
Once the matrix is fully populated, the ratio of data on- and off-diagonal values will be
evaluated according to:

- For elements on the diagonal: `confusion[i][i]/sum_j(confusion[j][j])>=min_diag`
- For elements off the diagonal: `confusion[j][i])/sum_j(confusion[j][i])<=max_off_diag`

Note that the rule has only one required parameter: `base_trial` the trial whose execution will invoke the rule.
It takes the following arguments:
- `base_trial`: The trial whose execution will invoke the rule
- `category_no`: (optional) Number of categories.
If not specified, the rule will take the larger of the number of categories in `labels` and `predictions`.
- `labels`: (optional, default `"labels"`) The tensor name for 1-d vector of true labels.
- `predictions`: (optoinal, default `"predictions"`) The tensor name for 1-d vector of estimated labels.
- `labels_collection`: (optional, default `"labels"`) The rule will inspect the tensors in this collection for `labels`.
- `predictions_collection`: (optional, default `"predictions"`) The rule will inspect the tensors in this collection for `predictions`.
- `min_diag`: (optional, default 0.9) Mininum value for the ratio of data on the diagonal.
- `max_off_diag`: (optional, default 0.1) Maximum value for the raio of data off the diagonal.

Note that this rule will infer the default parameters if configurations are not specified, so you can simply use
```python
rules_specification = [
    {
        "RuleName": "Confusion",
        "InstanceType": "ml.c5.4xlarge"
    }
]
```
If you want to specify the optional parameters, you can do so by using `RuntimeConfigurations`:
```python
rules_specification = [
    {
        "RuleName": "Confusion",
        "InstanceType": "ml.c5.4xlarge",
        "RuntimeConfigurations": {
            "category_no": "10",
            "min_diag": "0.8",
            "max_diag": "0.2"
        }
    }
]
```

This rule is applicable for using with XGBoost.
