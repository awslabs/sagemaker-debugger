#!/bin/bash

# To manually disable profiler integration tests from running in the PR CI, set this environment variable to "true".
# If you do this, remember to reset it back to "false" before merging the PR.
disable_integration_tests="false"
if [ $disable_integration_tests = "true" ]
then
  echo "PROFILER INTEGRATION TESTS MANUALLY DISABLED!"
  exit 0
fi

check_changed_files() {
  # Get the branch we're running integration tests on.
  export CODEBUILD_GIT_BRANCH=${CODEBUILD_WEBHOOK_HEAD_REF#refs/heads/};

  # If the branch is an empty string, that means the current branch is master and the integration tests will be run.
  if [[ $CODEBUILD_GIT_BRANCH = "" ]]; then
    echo "true"
    return
  fi

  # Otherwise, check to see what files have changed in this branch. If files in smdebug/core or smdebug/profiler or
  # smdebug/$framework have been modified, run the integration tests. Otherwise, don't run the integration tests.
  for file in $(git diff --name-only master $CODEBUILD_GIT_BRANCH)
  do
    root_folder=$(echo $file | cut -d/ -f 1)
    framework_folder=$(echo $file | cut -d/ -f 2)

    # Check if any relevant smdebug files were modified.
    if [ $root_folder = "smdebug" ] && [[ $framework_folder = "core" || $framework_folder = "profiler" || $framework_folder = "$framework" ]]; then
      echo "true"
      return
    fi

    # Check if relevant files for running profiler integration tests were modified.
    if [ $root_folder = "config" ] && [ $framework_folder = "profiler" ]; then
      echo "true"
      return
    fi

  done

  echo "false"
}

# If we are not force running the tests, determine whether to run the tests based on the files changed in the branch compared to master.
run_tests="true"
if [ $force_run_tests = "false" ]; then
  run_tests=$( check_changed_files )
fi

apt-get update >/dev/null 2>/dev/null # mask output
apt-get install sudo -qq -o=Dpkg::Use-Pty=0 >/dev/null 2>/dev/null # mask output
sudo apt-get install unzip -qq -o=Dpkg::Use-Pty=0 >/dev/null 2>/dev/null # mask output
pip install -q -r config/profiler/requirements.txt >/dev/null 2>/dev/null # mask output

cd $CODEBUILD_SRC_DIR
chmod +x config/protoc_downloader.sh
./config/protoc_downloader.sh >/dev/null 2>/dev/null # mask output

touch $CODEBUILD_SRC_DIR_TESTS/tests/scripts/tf_scripts/requirements.txt

if [ $framework = "pytorch" ]
then
  scripts_folder="pytorch_scripts"
  test_file="test_profiler_pytorch.py"
  # download the data from s3 bucket.
  cd $CODEBUILD_SRC_DIR_TESTS/tests/scripts/$scripts_folder
  mkdir -p data
  aws s3 cp s3://smdebug-testing/datasets/cifar-10-python.tar.gz data/cifar-10-batches-py.tar.gz >/dev/null 2>/dev/null # mask output
  aws s3 cp s3://smdebug-testing/datasets/MNIST_pytorch.tar.gz data/MNIST_pytorch.tar.gz >/dev/null 2>/dev/null # mask output
  cd $CODEBUILD_SRC_DIR_TESTS/tests/scripts/pytorch_scripts/data
  tar -zxf MNIST_pytorch.tar.gz >/dev/null 2>/dev/null # mask output
  tar -zxf cifar-10-batches-py.tar.gz >/dev/null 2>/dev/null # mask output
else
  scripts_folder="tf_scripts"
  test_file="test_profiler_tensorflow.py"
  echo "tensorflow-datasets==4.0.1" >> $CODEBUILD_SRC_DIR_TESTS/tests/scripts/tf_scripts/requirements.txt # Install tensorflow-datasets in container
fi

 # build pip wheel of the latest smdebug
cd $CODEBUILD_SRC_DIR
python setup.py bdist_wheel --universal >/dev/null 2>/dev/null
pip install -q --force-reinstall dist/*.whl >/dev/null 2>/dev/null # mask output

# install smdebug from current branch in the container or use the smdebug that's already in the container
if [ "$use_current_branch" = "true" ]; then
  cd $CODEBUILD_SRC_DIR/dist
  echo "./"`ls smdebug*` >> $CODEBUILD_SRC_DIR_TESTS/tests/scripts/$scripts_folder/requirements.txt
  cp smdebug*  $CODEBUILD_SRC_DIR_TESTS/tests/scripts/$scripts_folder/.
fi

if [ "$run_tests" = "true" ]
then
  # Run the smprofiler sagemaker integration tests
  cd $CODEBUILD_SRC_DIR_TESTS
  echo "Running profiler integration tests!"
  if python -m pytest -n auto -v -s -W=ignore --html=$REPORT_DIR/profiler_report_analysis.html --self-contained-html tests/$test_file
  then
    echo "INFO BUILD SUCCEEDED !!! "
    exit 0
  else
    echo "ERROR BUILD FAILED "
    exit 1
  fi
else
  echo "SKIPPING PROFILER INTEGRATION TESTS !!! "
  exit 0
fi
