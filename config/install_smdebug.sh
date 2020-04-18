#!/bin/bash
set -ex
set -o pipefail

CORE_REPO="https://github.com/awslabs/sagemaker-debugger.git"
RULES_REPO="https://$RULES_ACCESS_USER:$RULES_ACCESS_TOKEN@github.com/awslabs/sagemaker-debugger-rules.git"

if [ "$stable_release" = "enable" ]; then
  SMDEBUG_S3_BINARY="s3://smdebug-stable-release/$(date +%F)/";
elif [ "$stable_release" = "disable" ]; then
  SMDEBUG_S3_BINARY="s3://smdebug-nightly-binaries/$(date +%F)/";
fi

# Uninstall the built-in version of smdebug and assert that it no longer exists.
pip uninstall -y smdebug
#python -c "import smdebug"

code_dir=$(basename "$PWD")
echo "Cloning sagemaker-debugger repository."
cd $CODEBUILD_SRC_DIR && cd .. && rm -rf "$code_dir" && git clone "$CORE_REPO" "$code_dir"
echo "sagemaker-debugger repository cloned in the path $CODEBUILD_SRC_DIR"

cd $CODEBUILD_SRC_DIR_RULES && code_dir=$(basename "$PWD")
echo "Cloning sagemaker-debugger-rules repository."
cd $CODEBUILD_SRC_DIR_RULES && cd .. && rm -rf "$code_dir" && git clone "$RULES_REPO" "$code_dir"
echo "sagemaker-debugger-rules repository cloned in the path $CODEBUILD_SRC_DIR_RULES"
export RULES_CODEBUILD_SRC_DIR="$CODEBUILD_SRC_DIR_RULES"

export CODEBUILD_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export CODEBUILD_PROJECT=${CODEBUILD_BUILD_ID%:$CODEBUILD_LOG_PATH}
export CODEBUILD_BUILD_URL=https://$AWS_DEFAULT_REGION.console.aws.amazon.com/codebuild/home?region=$AWS_DEFAULT_REGION#/builds/$CODEBUILD_BUILD_ID/view/new
export CURRENT_DATETIME=$(date +'%Y%m%d_%H%M%S')
export CURRENT_COMMIT_PATH="$CURRENT_DATETIME/$CORE_COMMIT"

# you can provide pip binary as s3 path in the build environment
if [ "$SMDEBUG_S3_BINARY" ]; then
  cd $CODEBUILD_SRC_DIR
  echo "Installing smdebug and smdebug_rules from pre-generated pip wheels located at $SMDEBUG_S3_BINARY"
  mkdir -p s3_pip_binary
  aws s3 cp --recursive "$SMDEBUG_S3_BINARY" s3_pip_binary
  pip install --upgrade --force-reinstall s3_pip_binary/smdebug_rules-*.whl
  pip install --upgrade --force-reinstall s3_pip_binary/smdebug-*.whl
  export CORE_COMMIT=`cat s3_pip_binary/CORE_COMMIT`
  export RULES_COMMIT=`cat s3_pip_binary/RULES_COMMIT`
  echo "Commit hash on sagemaker-debugger-rules repository being used: $RULES_COMMIT"
  cd $CODEBUILD_SRC_DIR_RULES && git checkout "$RULES_COMMIT"
  python setup.py bdist_wheel --universal && pip install --force-reinstall dist/*.whl
  echo "Commit hash on sagemaker-debugger repository being used: $CORE_COMMIT"
  cd $CODEBUILD_SRC_DIR && git checkout "$CORE_COMMIT"
  python setup.py bdist_wheel --universal && pip install --force-reinstall dist/*.whl
else
  # if the env var stable_release is not set, then this else block is executed.
  if [ -z "$CORE_COMMIT" ]; then export CORE_COMMIT=$(git log -1 --pretty=%h); fi
  echo "Commit hash on sagemaker-debugger repository being used: $CORE_COMMIT"
  if [ -z "$RULES_COMMIT" ]; then export RULES_COMMIT=$(git log -1 --pretty=%h); fi
  echo "Commit hash on sagemaker-debugger-rules repository being used: $RULES_COMMIT"
  cd $CODEBUILD_SRC_DIR_RULES && git checkout "$RULES_COMMIT"  && python setup.py bdist_wheel --universal && pip install --force-reinstall dist/*.whl
  cd $CODEBUILD_SRC_DIR && git checkout "$CORE_COMMIT" && python setup.py bdist_wheel --universal && pip install --force-reinstall dist/*.whl
fi

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
