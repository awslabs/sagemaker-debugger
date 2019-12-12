# Examples
## Example notebooks
Please refer to the example notebooks in [Amazon SageMaker Examples repository](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-debugger)

## Example scripts
The above notebooks come with example scripts which can be used through SageMaker. Some more example scripts are here in [scripts/](scripts/)

## Example configurations for saving tensors through [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk)
Example configurations for saving tensors through the hook are available at [docs/sagemaker.md](../docs/sagemaker.md)

## Example configurations for running rules through [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk)
Example configurations for saving tensors through the hook are available at [docs/sagemaker.md](../docs/sagemaker.md)

## Example for running rule locally

```
from smdebug.rules import invoke_rule
from smdebug.trials import create_trial
trial = create_trial('s3://bucket/prefix')
rule_obj = CustomRule(trial, param=value)
invoke_rule(rule_obj, start_step=0, end_step=10)
```
