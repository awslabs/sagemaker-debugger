## Tornasole binaries
### Latest binaries
##### MXNet
```
s3://tornasole-external-preview-use1/sdk/ts-binaries/tornasole_mxnet/py3/latest/
```
##### TensorFlow
```
s3://tornasole-external-preview-use1/sdk/ts-binaries/tornasole_tensorflow/py3/latest/
```
##### PyTorch
```
s3://tornasole-external-preview-use1/sdk/ts-binaries/tornasole_pytorch/py3/latest/
```
##### XGBoost
```
s3://tornasole-external-preview-use1/sdk/ts-binaries/tornasole_xgboost/py3/latest/
```
##### Rules
```
s3://tornasole-external-preview-use1/sdk/ts-binaries/tornasole_rules/py3/latest/
```

### All versions
```
s3://tornasole-binaries-use1/tornasole_mxnet/py3/
s3://tornasole-binaries-use1/tornasole_tensorflow/py3/
s3://tornasole-binaries-use1/tornasole_pytorch/py3/
s3://tornasole-binaries-use1/tornasole_xgboost/py3/
s3://tornasole-binaries-use1/tornasole_rules/py3/
```

## Framework Containers with Tornasole
REGION below can be one of
```
us-east-1
us-east-2
us-west-1
us-west-2
ap-south-1
ap-northeast-2
ap-southeast-1
ap-southeast-2
ap-northeast-1
ca-central-1
eu-central-1
eu-west-1
eu-west-2
eu-west-3
eu-north-1
sa-east-1
```
#### TensorFlow
```
cpu	  072677473360.dkr.ecr.REGION.amazonaws.com/tornasole-preprod-tf-1.13.1-cpu:latest
gpu		072677473360.dkr.ecr.REGION.amazonaws.com/tornasole-preprod-tf-1.13.1-gpu:latest
```
#### MXNet
```
cpu		072677473360.dkr.ecr.REGION.amazonaws.com/tornasole-preprod-mxnet-1.4.1-cpu:latest
gpu		072677473360.dkr.ecr.REGION.amazonaws.com/tornasole-preprod-mxnet-1.4.1-gpu:latest
```
#### PyTorch
```
cpu		072677473360.dkr.ecr.REGION.amazonaws.com/tornasole-preprod-pytorch-1.1.0-cpu:latest
gpu		072677473360.dkr.ecr.REGION.amazonaws.com/tornasole-preprod-pytorch-1.1.0-gpu:latest
```
#### XGBoost
```
cpu		072677473360.dkr.ecr.REGION.amazonaws.com/tornasole-preprod-xgboost-0.90-cpu:latest
```


## Process for new release
- Tag the commit on alpha with the version you want.
- Create a branch for each minor version if it doesn't already exist (all 0.3.x will go into 0.3 branch). Push this new tag to that branch.
- For new features, use a new minor version. For patch releases, increment the patch version number (0.3.3->0.3.4).
- Update version in `tornasole/_version.py` in the release branch.
- Build new binaries
```
python bin/build_binaries.py --upload --replace-latest
```
- Build new containers
```
python bin/build_containers.py --tag $VERSION
```
- Test the new container
```
cd bin/sagemaker-containers/tensorflow
SM_TESTING_TAG=$VERSION python run_sagemaker.py
```
- When you are sure that the containers are good, then retag them as latest
```
bash tag_as_latest.sh $VERSION
```
