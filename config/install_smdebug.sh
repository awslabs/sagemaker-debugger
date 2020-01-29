#!/bin/bash
set -ex
set -o pipefail

CORE_REPO="https://github.com/awslabs/sagemaker-debugger.git"
RULES_REPO="https://$RULES_ACCESS_USER:$RULES_ACCESS_TOKEN@github.com/awslabs/sagemaker-debugger-rules.git"
SMDEBUG_S3_BINARY="s3://smdebug-nightly-binaries/$(date +%F)/"

# Uninstall the built-in version of smdebug and assert that it no longer exists.
pip uninstall -y smdebug
python -c "import smdebug"
if [ "$?" == 0 ]; then
  echo "Smdebug uninstall failed."
  exit 1
fi

echo "Cloning sagemaker-debugger repository."
cd && git clone "$CORE_REPO"
cd && export CODEBUILD_SRC_DIR=`pwd`/sagemaker-debugger
echo "sagemaker-debugger repository cloned in the path $CODEBUILD_SRC_DIR"

echo "Cloning sagemaker-debugger-rules repository."
cd && git clone "$RULES_REPO"
cd && export CODEBUILD_SRC_DIR_RULES=`pwd`/sagemaker-debugger-rules
echo "sagemaker-debugger-rules repository cloned in the path $CODEBUILD_SRC_DIR_RULES"

# you can provide pip binary as s3 path in the build environment
if [ "$SMDEBUG_S3_BINARY" ]; then
  cd $CODEBUILD_SRC_DIR
  echo "Installing smdebug and smdebug_rules from pre-generated pip wheels located at $SMDEBUG_S3_BINARY"
  mkdir -p s3_pip_binary
  aws s3 cp --recursive "$SMDEBUG_S3_BINARY" s3_pip_binary
  pip install --upgrade s3_pip_binary/*.whl
  CORE_COMMIT=`cat s3_pip_binary/CORE_COMMIT`
  RULES_COMMIT=`cat s3_pip_binary/RULES_COMMIT`
  echo "Commit hash on sagemaker-debugger-rules repository being used: $RULES_COMMIT"
  cd $CODEBUILD_SRC_DIR_RULES && git checkout "$RULES_COMMIT"
  echo "Commit hash on sagemaker-debugger repository being used: $CORE_COMMIT"
  cd $CODEBUILD_SRC_DIR && git checkout "$CORE_COMMIT"
  export CURRENT_DATETIME=$(date +'%Y%m%d_%H%M%S')
  export CURRENT_COMMIT_PATH="$CURRENT_DATETIME/$CORE_COMMIT"
else
  ./config/change_branch.sh
  cd $CODEBUILD_SRC_DIR && python setup.py bdist_wheel --universal && pip install --upgrade --force-reinstall dist/*.whl
  cd $CODEBUILD_SRC_DIR_RULES && python setup.py bdist_wheel --universal && pip install --upgrade --force-reinstall dist/*.whl && cd ..
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
