#!/usr/bin/env bash

set -ex
check_logs() {
  if grep -e "AssertionError" -e "Error" -e "ERROR" $1;
   then
    echo "Integration tests:FAILED."
    exit 1
  else
        echo "Integration tests: SUCCESS."
  fi
}

export TORNASOLE_LOG_LEVEL=debug
export BLOCK_STDOUT=TRUE
export BLOCK_STDERR=FALSE
python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/report_analysis.html --self-contained-html tests/analysis
python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/report_core.html --self-contained-html tests/core

if [ "$run_pytest_tensorflow" = "enable" ] ; then
    python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/report_tensorflow.html --self-contained-html tests/tensorflow
    python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/test_rules_tensorflow.html --self-contained-html -s tests/analysis/integration_testing_rules.py::test_test_rules --mode tensorflow --path_to_config ./tests/analysis/config.yaml 2>&1 | tee upload/$CURRENT_COMMIT_PATH/reports/test_rules_tensorflow.log
fi

if [ "$run_pytest_mxnet" = "enable" ] ; then
    python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/report_mxnet.html --self-contained-html tests/mxnet
    python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/test_rules_mxnet.html --self-contained-html -s tests/analysis/integration_testing_rules.py::test_test_rules --mode mxnet --path_to_config ./tests/analysis/config.yaml 2>&1 | tee upload/$CURRENT_COMMIT_PATH/reports/test_rules_mxnet.log
fi

if [ "$run_pytest_pytorch" = "enable" ] ; then
    python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/report_pytorch.html --self-contained-html tests/pytorch
    python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/test_rules_pytorch.html --self-contained-html -s tests/analysis/integration_testing_rules.py::test_test_rules --mode pytorch --path_to_config ./tests/analysis/config.yaml 2>&1 | tee upload/$CURRENT_COMMIT_PATH/reports/test_rules_pytorch.log
fi

check_logs upload/$CURRENT_COMMIT_PATH/reports/report_analysis.html
check_logs upload/$CURRENT_COMMIT_PATH/reports/report_core.html
check_logs upload/$CURRENT_COMMIT_PATH/reports/report_tensorflow.html
check_logs upload/$CURRENT_COMMIT_PATH/reports/report_mxnet.html
check_logs upload/$CURRENT_COMMIT_PATH/reports/report_pytorch.html
check_logs upload/$CURRENT_COMMIT_PATH/reports/test_rules_pytorch.log
check_logs upload/$CURRENT_COMMIT_PATH/reports/test_rules_mxnet.log
check_logs upload/$CURRENT_COMMIT_PATH/reports/test_rules_pytorch.log
