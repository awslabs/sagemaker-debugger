#!/usr/bin/env bash
# run from tornasole_core folder
set -ex

tag_and_push() {
  for REGION in us-east-1 us-east-2 us-west-1 us-west-2 ap-south-1 ap-northeast-2 ap-southeast-1 ap-southeast-2 ap-northeast-1 ca-central-1 eu-central-1 eu-west-1 eu-west-2 eu-west-3 eu-north-1 sa-east-1
  do
      $(aws ecr get-login --no-include-email --region $REGION)
      docker tag $ECR_REPO_NAME:$ECR_TAG_NAME 072677473360.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
      docker push 072677473360.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
  done
}

if [ -z "$1" ]; then echo "No tag passed" &&  exit 1; fi

export ECR_TAG_NAME=$1

pushd .
cd ~
rm -rf sagemaker-tensorflow-container
git clone https://github.com/aws/sagemaker-tensorflow-container.git
cd sagemaker-tensorflow-container
git checkout v2.0.7
python setup.py sdist
popd
cd bin/sagemaker-containers/tensorflow/1.14.0/
cp ~/sagemaker-tensorflow-container/dist/sagemaker_tensorflow_container-*.tar.gz .

rm -rf tornasole-binary/
export TORNASOLE_BINARY_PATH=s3://tornasole-binaries-use1/tornasole_tensorflow/py3/latest
aws s3 sync $TORNASOLE_BINARY_PATH tornasole-binary
cp tornasole-binary/*.whl .
export TORNASOLE_BINARY=`ls tornasole-*.whl`

export TF_ESTIMATOR_BINARY_LOCATION=https://tensorflow-aws.s3-us-west-2.amazonaws.com/1.14/Ubuntu/estimator/tensorflow_estimator-1.14.0-py2.py3-none-any.whl
curl -O $TF_ESTIMATOR_BINARY_LOCATION

build() {
    if [ -z "$1" ]; then echo "No mode passed" &&  exit 1; fi

    mode=$1

    export ECR_REPO_NAME=tornasole-preprod-tf-1.14.0-$mode
    # this is aws tf
    export TF_BINARY_LOCATION=https://tensorflow-aws.s3-us-west-2.amazonaws.com/1.14/AmazonLinux/$mode/latest-patch-latest-patch/tensorflow-1.14.0-cp36-cp36m-linux_x86_64.whl
    rm -rf tf-$mode.whl
    curl -o tf-$mode.whl $TF_BINARY_LOCATION
    docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME --build-arg py_version=3 \
        --build-arg framework_installable=tf-$mode.whl \
        --build-arg framework_support_installable=sagemaker_tensorflow_container-*.tar.gz \
        --build-arg tornasole_framework_installable=$TORNASOLE_BINARY \
        --build-arg tf_estimator_installable=`basename $TF_ESTIMATOR_BINARY_LOCATION` \
        -f Dockerfile.$mode .
    tag_and_push
}

build cpu &
build gpu &

wait
