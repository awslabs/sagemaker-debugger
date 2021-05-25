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
      python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  --durations=50 --html=$REPORT_DIR/report_$1.html -v -s --self-contained-html --ignore=tests/core/test_paths.py --ignore=tests/core/test_index_utils.py --ignore=tests/core/test_collections.py tests/$1
      if [ "$1" = "mxnet" ] ; then
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_mxnet_error_handling_agent.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_mxnet_gluon_integration.py
        # we run test/rules once, mxnet build has configured permission for sns to run this test
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/rules
      elif [ "$1" = "pytorch" ] ; then
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_pytorch_error_handling_agent.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_pytorch_integration.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_pytorch_multiprocessing.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_training_with_no_grad_updates.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_pytorch_disengage_smdebug.py
      elif [ "$1" = "tensorflow" ] ; then
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_tensorflow_integration.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_tensorflow_error_handling_agent.py
      elif [ "$1" = "tensorflow2" ] ; then
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_tensorflow2_error_handling_agent.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_tensorflow2_gradtape_integration.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_tensorflow2_integration.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_tensorflow2_bert.py
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append}  tests/zero_code_change/test_tensorflow_disengage_smdebug.py

      fi

    else
      if [ "$1" = "tensorflow2" ] ; then
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append} --durations=50 --html=$REPORT_DIR/report_$1/eager_mode.html -v -s --self-contained-html tests/$1
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append} --durations=50 --non-eager --html=$REPORT_DIR/report_$1/non_eager_mode.html -v -s --self-contained-html tests/$1
      else
        python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append} --durations=50 --html=$REPORT_DIR/report_$1.html -v -s --self-contained-html tests/$1
      fi
    fi
}

# Functionto invoke the profiler tests. The core tests in profiler are run for all the frameworks
run_profiler_test() {
    # Running the core tests in profiler.
    python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append} --durations=50 --html=$REPORT_DIR/report_profiler_core.html -v -s --self-contained-html tests/profiler/core

    # Running the framework specific profiler tests
    python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append} --durations=50 --html=$REPORT_DIR/report_profiler_$1.html -v -s --self-contained-html tests/profiler/$1
}
export TF_CPP_MIN_LOG_LEVEL=1
export SMDEBUG_LOG_LEVEL=info
#export BLOCK_STDOUT=TRUE
#export BLOCK_STDERR=FALSE

export OUT_DIR=upload/$CURRENT_COMMIT_PATH
export REPORT_DIR=$OUT_DIR/pytest_reports
python -m pytest ${code_coverage_smdebug:+--cov=./ --cov-append} -v -W=ignore --durations=50 --html=$REPORT_DIR/report_analysis.html --self-contained-html tests/analysis

run_for_framework core

if [ "$run_pytest_xgboost" = "enable" ] ; then
    run_for_framework xgboost
fi

if [ "$run_pytest_tensorflow" = "enable" ] ; then
    # tensorflow_datasets >= 4.0.0 requires tensorflow >= 2.1.0
    pip install tensorflow_datasets==3.2.1
    run_for_framework tensorflow
fi

if [ "$run_pytest_tensorflow2" = "enable" ] ; then
    pip install tensorflow_datasets==4.0.1
    run_for_framework tensorflow2
    run_profiler_test tensorflow2
fi

if [ "$run_pytest_mxnet" = "enable" ] ; then
    run_for_framework mxnet
fi

if [ "$run_pytest_pytorch" = "enable" ] ; then
    run_for_framework pytorch
    run_profiler_test pytorch
fi

check_logs $REPORT_DIR/*

# Only look at newly added files
if [ -n "$(git status --porcelain | grep ^?? | grep -v smdebugcodebuildtest | grep -v upload)" ]; then
    exit 0
fi
