# XGBoost

## Contents

- [SageMaker Example](#sagemaker-example)
- [Full API](#full-api)

## SageMaker Example

### Use XGBoost as a built-in algorithm

The XGBoost algorithm can be used 1) as a built-in algorithm, or 2) as a framework such as MXNet, PyTorch, or Tensorflow.
If SageMaker XGBoost is used as a built-in algorithm in container version `0.90-2` or later, Amazon SageMaker Debugger will be available by default (i.e., zero code change experience).
See [XGBoost Algorithm AWS docmentation](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) for more information on how to use XGBoost as a built-in algorithm.
See [Amazon SageMaker Debugger examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger) for sample notebooks that demonstrate debugging and monitoring capabilities of Amazon SageMaker Debugger.
See [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/) for more information on how to configure the Amazon SageMaker Debugger from the Python SDK.

### Use XGBoost as a framework

When SageMaker XGBoost is used as a framework, it is recommended that the hook is configured from the [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/).
By using SageMaker Python SDK, you can run different jobs (e.g., Processing jobs) on the SageMaker platform.
You can retrieve the hook as follows.
```python
import xgboost as xgb
from smdebug.xgboost import Hook

dtrain = xgb.DMatrix("train.libsvm")
dtest = xgb.DMatrix("test.libsmv")

hook = Hook.create_from_json_file()
hook.train_data = dtrain  # required
hook.validation_data = dtest  # optional
hook.hyperparameters = params  # optional

bst = xgb.train(
    params,
    dtrain,
    callbacks=[hook],
    evals_result=[(dtrain, "train"), (dvalid, "validation")]
)
```

Alternatively, you can also create the hook from `smdebug`'s Python API as shown in the next section.

### Use the Debugger hook

If you are in a non-SageMaker environment, or even in SageMaker, if you want to configure the hook in a certain way in script mode, you can use the full Debugger hook API as follows.
```python
import xgboost as xgb
from smdebug.xgboost import Hook

dtrain = xgb.DMatrix("train.libsvm")
dvalid = xgb.DMatrix("validation.libsmv")

hook = Hook(
    out_dir=out_dir,  # required
    train_data=dtrain,  # required
    validation_data=dvalid,  # optional
    hyperparameters=hyperparameters,  # optional
)
```

## Full API

```python
def __init__(
    self,
    out_dir,
    export_tensorboard = False,
    tensorboard_dir = None,
    dry_run = False,
    reduction_config = None,
    save_config = None,
    include_regex = None,
    include_collections = None,
    save_all = False,
    include_workers = "one",
    hyperparameters = None,
    train_data = None,
    validation_data = None,
)
```
Initializes the hook. Pass this object as a callback to `xgboost.train()`.
* `out_dir` (str): A path into which tensors and metadata will be written.
* `export_tensorboard` (bool): Whether to use TensorBoard logs.
* `tensorboard_dir` (str): Where to save TensorBoard logs.
* `dry_run` (bool): If true, evaluations are not actually saved to disk.
* `reduction_config` (ReductionConfig object): Not supported in XGBoost and will be ignored.
* `save_config` (SaveConfig object): See the [Common API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md).
* `include_regex` (list[str]): List of additional regexes to save.
* `include_collections` (list[str]): List of collections to save.
* `save_all` (bool): Saves all tensors and collections. **WARNING: May be memory-intensive and slow.**
* `include_workers` (str): Used for distributed training, can also be "all".
* `hyperparameters` (dict): Booster params.
* `train_data` (DMatrix object): Data to be trained.
* `validation_data` (DMatrix object): Validation set for which metrics will evaluated during training.

See the [Common API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md) page for details about Collection, SaveConfig, and ReductionConfig.\
See the [Analysis](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md) page for details about analyzing a training job.
