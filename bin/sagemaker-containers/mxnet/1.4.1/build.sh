#!/usr/bin/env bash
# run from tornasole_core folder
set -ex

tag_and_push() {
  for REGION in us-east-1 us-west-2
  do
      $(aws ecr get-login --no-include-email --region $REGION)
      docker tag $ECR_REPO_NAME:$ECR_TAG_NAME 072677473360.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
      docker push 072677473360.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
  done
}

export TORNASOLE_BINARY_PATH=s3://tornasole-binaries-use1/tornasole_mxnet/py3/tornasole-0.3-py2.py3-none-any.whl
export ECR_TAG_NAME=$1

cd bin/sagemaker-containers/mxnet/1.4.1/
aws s3 cp $TORNASOLE_BINARY_PATH .

export ECR_REPO_NAME=tornasole-preprod-mxnet-1.4.1-gpu

docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME \
    --build-arg tornasole_installable=`basename $TORNASOLE_BINARY_PATH` \
    -f Dockerfile.gpu .
tag_and_push

export ECR_REPO_NAME=tornasole-preprod-mxnet-1.4.1-cpu
docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME \
    --build-arg tornasole_installable=`basename $TORNASOLE_BINARY_PATH` \
    -f Dockerfile.cpu .
tag_and_push
