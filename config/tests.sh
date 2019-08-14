#!/usr/bin/env bash

#export TORNASOLE_LOG_LEVEL=debug
TORNASOLE_LOG_LEVEL=debug python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/report.html --self-contained-html tests/
TORNASOLE_LOG_LEVEL=debug python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/test_rules_tensorflow.html --self-contained-html -s tests/analysis/integration_testing_rules.py::test_test_rules --mode tensorflow --path_to_config ./tests/analysis/config.yaml
TORNASOLE_LOG_LEVEL=debug python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/test_rules_mxnet.html --self-contained-html -s tests/analysis/integration_testing_rules.py::test_test_rules --mode mxnet --path_to_config ./tests/analysis/config.yaml
TORNASOLE_LOG_LEVEL=debug python -m pytest --html=upload/$CURRENT_COMMIT_PATH/reports/test_rules_pytorch.html --self-contained-html -s tests/analysis/integration_testing_rules.py::test_test_rules --mode pytorch --path_to_config ./tests/analysis/config.yaml

