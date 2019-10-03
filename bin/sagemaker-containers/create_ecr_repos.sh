#!/usr/bin/env bash

for region in us-east-1 us-east-2 us-west-1 us-west-2 ap-south-1 ap-northeast-2 ap-southeast-1 ap-southeast-2 ap-northeast-1 ca-central-1 eu-central-1 eu-west-1 eu-west-2 eu-west-3 eu-north-1 sa-east-1
do
    for framework_version in tf-1.14.0 mxnet-1.4.1 pytorch-1.1.0
    do
        aws ecr --region $region create-repository --repository-name tornasole-preprod-$framework_version-cpu
        aws ecr --region $region create-repository --repository-name tornasole-preprod-$framework_version-gpu
        aws ecr --region $region set-repository-policy --repository-name tornasole-preprod-$framework_version-gpu --policy-text file://bin/sagemaker-containers/permissions.json
        aws ecr --region $region set-repository-policy --repository-name tornasole-preprod-$framework_version-cpu --policy-text file://bin/sagemaker-containers/permissions.json
    done
    aws ecr --region $region create-repository --repository-name tornasole-preprod-xgboost-0.90-cpu
    aws ecr --region $region set-repository-policy --repository-name tornasole-preprod-xgboost-0.90-cpu --policy-text file://bin/sagemaker-containers/permissions.json
done
