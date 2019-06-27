import boto3
import botocore
class SageMakerUtils:
    @staticmethod
    def is_sagemaker_job_finsihed(jobid, returnMock=None):
        if returnMock is not None:
            return returnMock
        client = boto3.client('sagemaker')
        response = client.describe_training_job(TrainingJobName=jobid)    
        status = response['TrainingJobStatus']
        if status in ['InProgress', 'Stopping']:
            return False
        elif status in ['Completed', 'Failed', 'Stopped']:
            return True  # return 1 if the job is finished
