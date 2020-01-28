#!/bin/bash
set -ex
set -o pipefail

CORE_REPO="https://github.com/awslabs/sagemaker-debugger.git"
RULES_REPO="https://github.com/awslabs/sagemaker-debugger-rules.git"
SMDEBUG_S3_BINARY="s3://smdebug-nightly-binaries/$(date +%F)/"

echo "Cloning sagemaker-debugger and sagemaker-debugger-rules repository."
cd
git clone "$CORE_REPO"
git clone "$RULES_REPO"

export CODEBUILD_SRC_DIR=`pwd`/sagemaker-debugger
export CODEBUILD_SRC_DIR_RULES=`pwd`/sagemaker-debugger-rules

echo "sagemaker-debugger repository cloned in the path $CODEBUILD_SRC_DIR"
echo "sagemaker-debugger-rules repository cloned in the path $CODEBUILD_SRC_DIR_RULES"

cd $CODEBUILD_SRC_DIR
echo "Installing smdebug and smdebug_rules from pre-generated pip wheels at $SMDEBUG_S3_BINARY"
mkdir -p s3_pip_binary
aws s3 cp --recursive "$SMDEBUG_S3_BINARY" s3_pip_binary
pip install --upgrade s3_pip_binary/*.whl

CORE_COMMIT=`cat s3_pip_binary/CORE_COMMIT`
RULES_COMMIT=`cat s3_pip_binary/RULES_COMMIT`

echo "Commit hash on sagemaker-debugger repository being used: $CORE_COMMIT"
echo "Commit hash on sagemaker-debugger-rules repository being used: $RULES_COMMIT"

cd $CODEBUILD_SRC_DIR_RULES && git checkout "$RULES_COMMIT"

cd $CODEBUILD_SRC_DIR && git checkout "$CORE_COMMIT"

if [ "$run_pytest_mxnet" == 'enable' ]; then
  ./config/check_smdebug_install.sh mxnet
fi
if [ "$run_pytest_tensorflow" == 'enable' ]; then
  ./config/check_smdebug_install.sh tensorflow
  pip install tensorflow_datasets
fi
if [ "$run_pytest_pytorch" == 'enable' ]; then
  ./config/check_smdebug_install.sh torch
fi
if [ "$run_pytest_xgboost" == 'enable' ]; then
  ./config/check_smdebug_install.sh xgboost
  pip install --ignore-installed PyYAML
fi
