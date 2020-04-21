#!/usr/bin/env bash

set -e
set -ex
set -o pipefail

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
    if [ "$zero_code_change_test" = "enable" ] ; then
      # ignoring some test becuase they require multiple frmaeworks to be installed, these tests need to be broken down
      python -m pytest --durations=50 --html=$REPORT_DIR/report_$1.html -v -s --self-contained-html --ignore=tests/core/test_paths.py --ignore=tests/core/test_index_utils.py --ignore=tests/core/test_collections.py tests/$1
      if [ "$1" = "mxnet" ] ; then
        python tests/zero_code_change/mxnet_gluon_integration_test.py
      elif [ "$1" = "pytorch" ] ; then
        python tests/zero_code_change/pytorch_integration_tests.py
      elif [ "$1" = "tensorflow" ] ; then
        python tests/zero_code_change/tensorflow_integration_tests.py
      elif [ "$1" = "tensorflow2" ] ; then
        python tests/zero_code_change/tensorflow2_gradtape_integration_tests.py
        python tests/zero_code_change/tensorflow2_integration_tests.py
      fi

    else
      if [ "$1" = "tensorflow2" ] ; then
        python -m pytest --durations=50 --html=$REPORT_DIR/report_$1/eager_mode.html -v -s --self-contained-html tests/$1
        python -m pytest --durations=50 --non-eager --html=$REPORT_DIR/report_$1/non_eager_mode.html -v -s --self-contained-html tests/$1
      else
        python -m pytest --durations=50 --html=$REPORT_DIR/report_$1.html -v -s --self-contained-html tests/$1
      fi
    fi
}

export TF_CPP_MIN_LOG_LEVEL=1
export SMDEBUG_LOG_LEVEL=info
#export BLOCK_STDOUT=TRUE
#export BLOCK_STDERR=FALSE

export OUT_DIR=upload/$CURRENT_COMMIT_PATH
export REPORT_DIR=$OUT_DIR/pytest_reports
python -m pytest -v -W=ignore --durations=50 --html=$REPORT_DIR/report_analysis.html --self-contained-html tests/analysis

run_for_framework core

if [ "$run_pytest_xgboost" = "enable" ] ; then
    run_for_framework xgboost
fi

if [ "$run_pytest_tensorflow" = "enable" ] ; then
    run_for_framework tensorflow
fi

if [ "$run_pytest_tensorflow2" = "enable" ] ; then
    run_for_framework tensorflow2
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
    exit 0
fi
