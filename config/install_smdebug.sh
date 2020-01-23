#!/bin/bash
set -ex
set -o pipefail


export SMDEBUG_S3_BINARY="s3://smdebug-nightly-binaries/$(date +%F)/"
cd $CODEBUILD_SRC_DIR
# you can provide bip binary as s3 path in the build environment
if [ "$SMDEBUG_S3_BINARY" ]; then
  echo "Installing smdebug and smdebug_rules from pre-generated pip wheels at $SMDEBUG_S3_BINARY"
  mkdir -p s3_pip_binary
  aws s3 cp --recursive "$SMDEBUG_S3_BINARY" s3_pip_binary
  pip install --upgrade s3_pip_binary/*.whl
else
  python setup.py bdist_wheel --universal && pip install --upgrade --force-reinstall dist/*.whl
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
