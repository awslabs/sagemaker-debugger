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

export ECR_TAG_NAME=$1

pushd .
cd ~
rm -rf sagemaker-pytorch-container
git clone https://github.com/aws/sagemaker-pytorch-container.git
cd sagemaker-pytorch-container
python setup.py bdist_wheel
popd
cd bin/sagemaker-containers/pytorch/1.1.0/
cp ~/sagemaker-pytorch-container/dist/sagemaker_pytorch_container-*.whl .
cp -r ~/sagemaker-pytorch-container/lib .

export TORNASOLE_BINARY_PATH=s3://tornasole-binaries-use1/tornasole_mxnet/py3/latest
aws s3 sync $TORNASOLE_BINARY_PATH tornasole-binary
cp tornasole-binary/*.whl .
export TORNASOLE_BINARY=`ls tornasole-*.whl`

export ECR_REPO_NAME=tornasole-preprod-pytorch-1.1.0-gpu
docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME --build-arg py_version=3 \
    --build-arg tornasole_framework_installable=$TORNASOLE_BINARY \
    -f Dockerfile.gpu .
tag_and_push

export ECR_REPO_NAME=tornasole-preprod-pytorch-1.1.0-cpu
docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME --build-arg py_version=3 \
    --build-arg tornasole_framework_installable=$TORNASOLE_BINARY \
    -f Dockerfile.cpu .
tag_and_push