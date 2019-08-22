#!/usr/bin/env bash

for region in us-east-1 us-east-2 us-west-1 us-west-2 ap-south-1 ap-northeast-2 ap-southeast-1 ap-southeast-2 ap-northeast-1 ca-central-1 eu-central-1 eu-west-1 eu-west-2 eu-west-3 eu-north-1 sa-east-1
do
    for framework_version in tf-1.13.1 mxnet-1.4.1 pytorch-1.1.0
    do
        aws ecr --region $region create-repository --repository-name tornasole-preprod-$framework_version-cpu
        aws ecr --region $region create-repository --repository-name tornasole-preprod-$framework_version-gpu
    done
done