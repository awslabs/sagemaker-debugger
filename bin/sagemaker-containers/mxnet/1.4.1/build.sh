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

cd bin/sagemaker-containers/mxnet/1.4.1/

export SMDEBUG_BINARY_PATH=s3://tornasole-binaries-use1/tornasole_mxnet/py3/latest
rm -rf tornasole-binary/
aws s3 sync $SMDEBUG_BINARY_PATH tornasole-binary
cp tornasole-binary/*.whl .
export TORNASOLE_BINARY=`ls tornasole-*.whl`

build() {
    if [ -z "$1" ]; then echo "No mode passed" &&  exit 1; fi
    mode=$1

    export ECR_REPO_NAME=tornasole-preprod-mxnet-1.4.1-$mode
    docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME \
        --build-arg tornasole_installable=$TORNASOLE_BINARY \
        -f Dockerfile.$mode .
    tag_and_push
}

build cpu &
build gpu &

wait
