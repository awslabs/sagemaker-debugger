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
rm -rf sagemaker-tensorflow-container
git clone https://github.com/aws/sagemaker-tensorflow-container.git
cd sagemaker-tensorflow-container
git checkout script-mode
python setup.py sdist
popd
cd bin/sagemaker-containers/tensorflow/1.13.1/
cp ~/sagemaker-tensorflow-container/dist/sagemaker_tensorflow_container-*.tar.gz .

export TORNASOLE_BINARY_PATH=s3://tornasole-binaries-use1/tornasole_mxnet/py3/latest
aws s3 sync $TORNASOLE_BINARY_PATH tornasole-binary
cp tornasole-binary/*.whl .
export TORNASOLE_BINARY=`ls tornasole-*.whl`

export TF_ESTIMATOR_BINARY_LOCATION=https://tensorflow-aws.s3-us-west-2.amazonaws.com/1.13/Ubuntu/estimator/tensorflow_estimator-1.13.0-py2.py3-none-any.whl
curl -O $TF_ESTIMATOR_BINARY_LOCATION

##### GPU

export ECR_REPO_NAME=tornasole-preprod-tf-1.13.1-gpu

# this is for pip tf
#export TF_BINARY_LOCATION=https://files.pythonhosted.org/packages/7b/b1/0ad4ae02e17ddd62109cd54c291e311c4b5fd09b4d0678d3d6ce4159b0f0/tensorflow_gpu-1.13.1-cp36-cp36m-manylinux1_x86_64.whl
# this is aws tf
export TF_BINARY_LOCATION=https://tensorflow-aws.s3-us-west-2.amazonaws.com/1.13/AmazonLinux/gpu/latest-patch-latest-patch/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl

curl -O $TF_BINARY_LOCATION
docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME --build-arg py_version=3 \
    --build-arg framework_installable=`basename $TF_BINARY_LOCATION` \
    --build-arg framework_support_installable=sagemaker_tensorflow_container-*.tar.gz \
    --build-arg tornasole_framework_installable=$TORNASOLE_BINARY \
    --build-arg tf_estimator_installable=`basename $TF_ESTIMATOR_BINARY_LOCATION` \
    -f Dockerfile.gpu .
tag_and_push

##### CPU
export ECR_REPO_NAME=tornasole-preprod-tf-1.13.1-cpu
# this is pip TF
#export TF_BINARY_LOCATION=https://files.pythonhosted.org/packages/77/63/a9fa76de8dffe7455304c4ed635be4aa9c0bacef6e0633d87d5f54530c5c/tensorflow-1.13.1-cp36-cp36m-manylinux1_x86_64.whl
# this is AWS TF
export TF_BINARY_LOCATION=https://tensorflow-aws.s3-us-west-2.amazonaws.com/1.13/AmazonLinux/cpu/latest-patch-latest-patch/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl

curl -O $TF_BINARY_LOCATION
docker build -t $ECR_REPO_NAME:$ECR_TAG_NAME --build-arg py_version=3 \
    --build-arg framework_installable=`basename $TF_BINARY_LOCATION` \
    --build-arg framework_support_installable=sagemaker_tensorflow_container-*.tar.gz \
    --build-arg tornasole_framework_installable=$TORNASOLE_BINARY \
    --build-arg tf_estimator_installable=`basename $TF_ESTIMATOR_BINARY_LOCATION` \
    -f Dockerfile.cpu .
tag_and_push
