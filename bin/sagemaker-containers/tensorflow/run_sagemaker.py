from sagemaker.tensorflow import TensorFlow
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
train_script_path = os.path.join(dir_path, 'tf-train.py')

tag = os.environ.get('SM_TESTING_TAG', 'DEFAULTTAGWHICHWILLFAIL')

estimator = TensorFlow(entry_point=train_script_path,
                       image_name='072677473360.dkr.ecr.us-east-1.amazonaws.com/tornasole-preprod-tf-1.13.1-cpu:' + tag,
                       role='AmazonSageMaker-ExecutionRole-20190614T145575', # hardcode role name
                       base_job_name='tornasole', #there are some restrictions on base job name so keep it simple
                       train_instance_count=1,
                       py_version='py3',
                       framework_version='1.13',
                       train_instance_type='ml.m4.xlarge')
estimator.fit()

estimator = TensorFlow(entry_point=train_script_path,
                       image_name='072677473360.dkr.ecr.us-east-1.amazonaws.com/tornasole-preprod-tf-1.13.1-gpu:' + tag,
                       role='AmazonSageMaker-ExecutionRole-20190614T145575', # hardcode role name
                       base_job_name='tornasole', #there are some restrictions on base job name so keep it simple
                       train_instance_count=1,
                       py_version='py3',
                       framework_version='1.13',
                       train_instance_type='ml.p2.xlarge')
estimator.fit()
