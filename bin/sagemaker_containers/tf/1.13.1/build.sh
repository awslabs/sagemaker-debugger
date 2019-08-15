#!/usr/bin/env bash
# run from tornasole_core folder
set -ex


export TORNASOLE_BINARY_PATH=s3://tornasole-binaries-use1/tornasole_tensorflow/py3/tornasole-0.3-py2.py3-none-any.whl
export TORNASOLE_BINARY_NAME=tornasole-0.3-py2.py3-none-any.whl
export ECR_TAG_NAME=latest

pushd .
cd ~
git clone https://github.com/aws/sagemaker-tensorflow-container.git
cd sagemaker-tensorflow-container
git checkout script-mode
python setup.py sdist
popd
cd bin/sagemaker_containers/tf/1.13.1/
cp ~/sagemaker-tensorflow-container/dist/sagemaker_tensorflow_container-*.tar.gz .
aws s3 cp $TORNASOLE_BINARY_PATH .

export ECR_REPO_NAME=tornasole-preprod-tf-1.13.1-gpu
export TF_BINARY_LOCATION=https://files.pythonhosted.org/packages/7b/b1/0ad4ae02e17ddd62109cd54c291e311c4b5fd09b4d0678d3d6ce4159b0f0/tensorflow_gpu-1.13.1-cp36-cp36m-manylinux1_x86_64.whl
export TF_BINARY_NAME=tensorflow_gpu-1.13.1-cp36-cp36m-manylinux1_x86_64.whl

#curl -O $TF_BINARY_LOCATION
#$(aws ecr get-login --no-include-email --region us-east-1)
#docker build -t $ECR_REPO_NAME --build-arg py_version=3 \
#    --build-arg framework_installable=$TF_BINARY_NAME \
#    --build-arg framework_support_installable=sagemaker_tensorflow_container-*.tar.gz \
#    --build-arg tornasole_framework_installable=$TORNASOLE_BINARY_NAME \
#    -f Dockerfile.gpu .
#docker tag $ECR_REPO_NAME:$ECR_TAG_NAME 072677473360.dkr.ecr.us-east-1.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
#docker push 072677473360.dkr.ecr.us-east-1.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
#
#export ECR_REPO_NAME=tornasole-preprod-tf-1.13.1-cpu
#export TF_BINARY_LOCATION=https://files.pythonhosted.org/packages/77/63/a9fa76de8dffe7455304c4ed635be4aa9c0bacef6e0633d87d5f54530c5c/tensorflow-1.13.1-cp36-cp36m-manylinux1_x86_64.whl
#export TF_BINARY_NAME=tensorflow-1.13.1-cp36-cp36m-manylinux1_x86_64.whl

curl -O $TF_BINARY_LOCATION
$(aws ecr get-login --no-include-email --region us-east-1)
docker build -t $ECR_REPO_NAME --build-arg py_version=3 \
    --build-arg framework_installable=$TF_BINARY_NAME \
    --build-arg framework_support_installable=sagemaker_tensorflow_container-*.tar.gz \
    --build-arg tornasole_framework_installable=$TORNASOLE_BINARY_NAME \
    -f Dockerfile.gpu .
docker tag $ECR_REPO_NAME:$ECR_TAG_NAME 072677473360.dkr.ecr.us-east-1.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
docker push 072677473360.dkr.ecr.us-east-1.amazonaws.com/$ECR_REPO_NAME:$ECR_TAG_NAME
