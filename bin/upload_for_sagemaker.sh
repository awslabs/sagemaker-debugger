#!/usr/bin/env bash

# deliberately export a bad profile so you don't run this script unless you really want to
export AWS_PROFILE=removethissoitdoesntcrash

# API DOCS
aws s3 cp docs/mxnet/api.md s3://tornasole-external-preview-use1/frameworks/mxnet/
aws s3 cp docs/tensorflow/api.md s3://tornasole-external-preview-use1/frameworks/tensorflow/
aws s3 cp docs/pytorch/api.md s3://tornasole-external-preview-use1/frameworks/pytorch/

# DEV GUIDES
aws s3 cp sagemaker-docs/DeveloperGuide_MXNet.md s3://tornasole-external-preview-use1/frameworks/mxnet/
aws s3 cp sagemaker-docs/DeveloperGuide_TF.md s3://tornasole-external-preview-use1/frameworks/tensorflow/
aws s3 cp sagemaker-docs/DeveloperGuide_PyTorch.md s3://tornasole-external-preview-use1/frameworks/pytorch/
aws s3 cp sagemaker-docs/DeveloperGuide_Rules.md s3://tornasole-external-preview-use1/rules/

# MXNET EXAMPLES
aws s3 sync examples/mxnet/sagemaker-notebooks s3://tornasole-external-preview-use1/frameworks/mxnet/examples/notebooks
aws s3 cp examples/mxnet/scripts/mnist_mxnet.py s3://tornasole-external-preview-use1/frameworks/mxnet/examples/scripts

# TF EXAMPLES
aws s3 sync examples/tensorflow/sagemaker-notebooks s3://tornasole-external-preview-use1/frameworks/tensorflow/examples/notebooks
aws s3 cp examples/tensorflow/scripts/simple.py s3://tornasole-external-preview-use1/frameworks/tensorflow/examples/scripts

# PYTORCH EXAMPLES
#aws s3 sync examples/pytorch s3://tornasole-external-preview-use1/frameworks/pytorch/examples

# RULES EXAMPLES
aws s3 cp "examples/rules/Bring Your Own Rule.ipynb" s3://tornasole-external-preview-use1/rules/

# RULES_PACKAGE
aws s3 sync tornasole/rules s3://tornasole-external-preview-use1/rules/rules_package

# BINARY
aws s3 sync --delete s3://tornasole-binaries-use1/tornasole_rules/py3/latest s3://tornasole-external-preview-use1/rules/binary

