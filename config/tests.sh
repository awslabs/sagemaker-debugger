#!/usr/bin/env bash

set -ex

check_logs() {
  if grep -e "AssertionError" $1;
   then
    echo "Integration tests: FAILED."
    exit 1
  else
        echo "Integration tests: SUCCESS."
  fi
}

run_for_framework() {
    python -m pytest --html=$REPORT_DIR/report_$1.html -v -s --self-contained-html tests/$1
}

export TF_CPP_MIN_LOG_LEVEL=1
export SMDEBUG_LOG_LEVEL=info
#export BLOCK_STDOUT=TRUE
#export BLOCK_STDERR=FALSE

export OUT_DIR=upload/$CURRENT_COMMIT_PATH
export REPORT_DIR=$OUT_DIR/pytest_reports
python -m pytest -v -W=ignore --html=$REPORT_DIR/report_analysis.html --self-contained-html tests/analysis
python -m pytest -v -W=ignore --html=$REPORT_DIR/report_core.html --self-contained-html tests/core

if [ "$run_pytest_xgboost" = "enable" ] ; then
    run_for_framework xgboost
fi

if [ "$run_pytest_tensorflow" = "enable" ] ; then
    run_for_framework tensorflow
fi

if [ "$run_pytest_mxnet" = "enable" ] ; then
    run_for_framework mxnet
fi

if [ "$run_pytest_pytorch" = "enable" ] ; then
    run_for_framework pytorch
fi

check_logs $REPORT_DIR/*

# Only look at newly added files
if [ -n "$(git status --porcelain | grep ^?? | grep -v smdebugcodebuildtest | grep -v upload)" ]; then
  echo "ERROR: Test artifacts were created. Please place these in /tmp."
  exit 1
fi
