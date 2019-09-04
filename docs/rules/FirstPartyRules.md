# First Party Rules

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

### ExplodingTensor
This rule helps you identify if you are running into a situation where any tensor has non finite values.

Note that it takes two parameters, `base_trial` the trial whose execution will invoke the rule and
`only_nan` which can be set to True if you only want to monitor for `nan` and not for `infinity`. By default, `only_nan` is set to False, which means it treats `nan` and `infinity` as exploding. 
                        
Here's how you import and instantiate this rule. 

```
from tornasole.rules.generic import ExplodingTensor
r = ExplodingTensor(base_trial)
```

### SimilarAcrossRuns
This rule helps you compare tensors across runs. Note that this rule takes two trials as inputs. First trial is the `base_trial` whose
execution will invoke the rule, and the `other_trial` is what is used to compare this trial's tensors with. 
The third argument is a regex pattern which can be used to restrict this comparision to certain tensors. If this is not passed, it includes all tensors by default.

It returns `True` if tensors are different at a given step between the two trials.

```
from tornasole.rules.generic import SimilarAcrossRuns
r = SimilarAcrossRuns(base_trial, other_trial, include=None)
```

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

```
from tornasole.rules.generic import WeightUpdateRatio
wur = WeightUpdateRatio(base_trial, large_threshold, small_threshold)
```

### AllZero
This rule helps to identify whether the tensors contain all zeros. It takes following arguments

- `base_trial`: The trial whose execution will invoke the rule. The rule will inspect the tensors gathered during this trial.
- `collection_names`: The list of collection names. The rule will inspect the tensors that belong to these collections.
- `tensor_regex`: The list of regex patterns. The rule will inspect the tensors that match the regex patterns specified in this list.

For this rule, users must specify either the `collection_names` or `tensor_regex` parameter. If both the parameters are specified the rule will inspect union on tensors.

```
from tornasole.rules.generic import AllZero
collections = ['weights', 'bias']
tensor_regex = ['input*']
allzero = AllZero(base_trial=trial_obj, collection_names=collections, tensor_regex=tensor_regex)
```

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

```
from tornasole.rules.generic import UnchangedTensor
ut = UnchangedTensor(base_trial=trial_obj, tensor_regex=['.*'], num_steps=3)
```

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

```
from tornasole.rules.generic import LossNotDecreasing
lnd = LossNotDecreasing(base_trial=trial_obj, tensor_regex=['loss*'], num_steps=20)
```
