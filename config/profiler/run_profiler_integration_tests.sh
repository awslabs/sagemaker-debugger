# If we are not force running the tests, determine whether to run the tests based on the files changed in the branch compared to master.
if [ "$force_run_tests" = "true" ]
then
  run_tests="true"
else
  chmod +x config/profiler/check_changed_files.sh
  run_tests=`./config/check_changed_files.sh $framework`
fi

apt-get update
apt-get install sudo -qq -o=Dpkg::Use-Pty=0
sudo apt-get install unzip -qq -o=Dpkg::Use-Pty=0

pip install -r config/profiler/requirements.txt

cd $CODEBUILD_SRC_DIR
chmod +x config/protoc_downloader.sh
./config/protoc_downloader.sh

touch $CODEBUILD_SRC_DIR_TESTS/tests/scripts/tf_scripts/requirements.txt

if [ $framework = "pytorch" ]
then
  scripts_folder="pytorch_scripts"
  test_file="test_profiler_pytorch.py"
  # download the data from s3 bucket.
  cd $CODEBUILD_SRC_DIR_TESTS/tests/scripts/$scripts_folder
  mkdir -p data
  aws s3 cp s3://smdebug-testing/datasets/cifar-10-python.tar.gz data/cifar-10-batches-py.tar.gz
  aws s3 cp s3://smdebug-testing/datasets/MNIST_pytorch.tar.gz data/MNIST_pytorch.tar.gz
  cd $CODEBUILD_SRC_DIR_TESTS/tests/scripts/pytorch_scripts/data
  cd data
  tar -zxf MNIST_pytorch.tar.gz
  tar -zxf cifar-10-batches-py.tar.gz
else
  scripts_folder="tf_scripts"
  test_file="test_profiler_tensorflow.py"
  echo "tensorflow-datasets==4.0.1" >> $CODEBUILD_SRC_DIR_TESTS/tests/scripts/tf_scripts/requirements.txt
fi

 # build pip wheel of the latest smdebug
cd $CODEBUILD_SRC_DIR
python setup.py bdist_wheel --universal
pip install --force-reinstall dist/*.whl

echo "horovod==0.19.5" >> $CODEBUILD_SRC_DIR_TESTS/tests/scripts/$scripts_folder/requirements.txt  # TODO: remove after fixing https://sim.amazon.com/issues/P42199318

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
