#!/usr/bin/env bash

# deliberately export a bad profile so you don't run this script unless you really want to
export AWS_PROFILE=removethissoitdoesntcrash

# API DOCS
aws s3 cp docs/mxnet/api.md s3://tornasole-external-preview-use1/frameworks/mxnet/
aws s3 cp docs/tensorflow/api.md s3://tornasole-external-preview-use1/frameworks/tensorflow/
aws s3 cp docs/tensorflow/examples/sm_resnet50.md s3://tornasole-external-preview-use1/frameworks/tensorflow/
aws s3 cp docs/pytorch/api.md s3://tornasole-external-preview-use1/frameworks/pytorch/

# DEV GUIDES
aws s3 cp sagemaker-docs/DeveloperGuide_MXNet.md s3://tornasole-external-preview-use1/frameworks/mxnet/
aws s3 cp sagemaker-docs/DeveloperGuide_TF.md s3://tornasole-external-preview-use1/frameworks/tensorflow/
aws s3 cp sagemaker-docs/DeveloperGuide_PyTorch.md s3://tornasole-external-preview-use1/frameworks/pytorch/
aws s3 cp sagemaker-docs/DeveloperGuide_Rules.md s3://tornasole-external-preview-use1/rules/
aws s3 cp sagemaker-docs/FirstPartyRules.md s3://tornasole-external-preview-use1/rules/

# MXNET EXAMPLES
aws s3 sync examples/mxnet/sagemaker-notebooks s3://tornasole-external-preview-use1/frameworks/mxnet/examples/notebooks
aws s3 sync examples/mxnet/scripts s3://tornasole-external-preview-use1/frameworks/mxnet/examples/scripts

# TF EXAMPLES
aws s3 sync examples/tensorflow/sagemaker-notebooks s3://tornasole-external-preview-use1/frameworks/tensorflow/examples/notebooks
aws s3 sync examples/tensorflow/scripts s3://tornasole-external-preview-use1/frameworks/tensorflow/examples/scripts

# PYTORCH EXAMPLES
aws s3 sync examples/pytorch/sagemaker-notebooks s3://tornasole-external-preview-use1/frameworks/pytorch/examples/notebooks
aws s3 cp examples/pytorch/scripts/simple.py s3://tornasole-external-preview-use1/frameworks/pytorch/examples/scripts/simple.py

# RULES EXAMPLES
aws s3 cp examples/rules/sagemaker-notebooks/BringYourOwnRule.ipynb s3://tornasole-external-preview-use1/rules/notebooks/
aws s3 cp examples/rules/scripts/my_custom_rule.py s3://tornasole-external-preview-use1/rules/scripts/my_custom_rule.py

# RULES_PACKAGE
aws s3 sync tornasole/rules s3://tornasole-external-preview-use1/rules/rules_package


