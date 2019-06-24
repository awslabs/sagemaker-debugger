import boto3
class SageMakerUtils:
    @staticmethod
    def is_sagemaker_job_finsihed(jobid, returnMock=None):
        if returnMock is not None:
            return returnMock
        try:
            client = boto3.client('sagemaker')
        except botocore.exceptions.UnknownServiceError:
            raise
        
        try:
            response = client.describe_training_job(TrainingJobName=jobid)            
            status = response['TrainingJobStatus']
            if status in ['InProgress', 'Stopping']:
                return False
            elif status in ['Completed', 'Failed', 'Stopped']:
                return True  # return 1 if the job is finished
        except botocore.exceptions.ClientError:
            raise
        except botocore.exceptions.EndpointConnectionError:
            raise
