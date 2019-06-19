import boto3
import time
import botocore


class CheckStatus:

    def __init__(self, aws_access_key_id, aws_secret_access_key, aws_session_token, JobId, region, interval):
        self.access_key = aws_access_key_id
        self.secret_key = aws_secret_access_key
        self.session_token = aws_session_token
        self.jobid = JobId
        self.service = 'sagemaker'
        self.region_name = region
        self.itv = interval

    def check_training_status(self):
        try:
            client = boto3.client(self.service, region_name=self.region_name,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                aws_session_token=self.session_token
            )
        except botocore.exceptions.UnknownServiceError:
            raise
        else:
            try:
                response = client.describe_training_job(TrainingJobName=self.jobid)
                # response = client.describe_notebook_instance(NotebookInstanceName=self.jobid)
            except botocore.exceptions.ClientError:
                raise
            except botocore.exceptions.EndpointConnectionError:
                raise
            else:
                status = response['TrainingJobStatus']
                if status in ['InProgress', 'Stopping']:
                    return False
                elif status in ['Completed', 'Failed', 'Stopped']:
                    return True  # return 1 if the job is finished

# c = CheckStatus(ACCESS_KEY, SECRET_KEY, SESSION_TOKEN, JobId, region)
# c.check_training_status()








