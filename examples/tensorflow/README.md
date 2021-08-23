# Examples
## Notebooks
Please refer to the example notebooks in the [Amazon SageMaker examples repository](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger).

## Scripts
The example notebooks in the repository linked in the preceding section come with example scripts which can be used through SageMaker. For more example scripts, see the [scripts/](scripts/) directory.

## Saving tensors through the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk)
For example configurations for saving tensors through the hook, see [docs/sagemaker.md](../docs/sagemaker.md).

## Running rules through the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk)
For example configurations for saving tensors through the hook, see [docs/sagemaker.md](../docs/sagemaker.md).

## Running rules locally

```
from smdebug.rules import invoke_rule
from smdebug.trials import create_trial
trial = create_trial('s3://bucket/prefix')
rule_obj = CustomRule(trial, param=value)
invoke_rule(rule_obj, start_step=0, end_step=10)
```
