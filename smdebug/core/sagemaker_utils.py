# Standard Library
import os

# Third Party
import boto3

# First Party
from smdebug.core.config_constants import DEFAULT_SAGEMAKER_OUTDIR


def is_sagemaker_job():
    """
    If this variable is defined we are assuming that this is
    a Sagemaker job. This is guaranteed to be defined
    for all Sagemaker jobs.
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-environment-variables
    :return: True or False
    """
    return "TRAINING_JOB_NAME" in os.environ


def get_sagemaker_out_dir():
    return DEFAULT_SAGEMAKER_OUTDIR


class SageMakerUtils:
    @staticmethod
    def is_sagemaker_job_finished(jobname, returnMock=None):
        if returnMock is not None:
            return returnMock
        client = boto3.client("sagemaker")
        response = client.describe_training_job(TrainingJobName=jobname)
        status = response["TrainingJobStatus"]
        if status in ["InProgress", "Stopping"]:
            return False
        elif status in ["Completed", "Failed", "Stopped"]:
            return True  # return 1 if the job is finished

    @staticmethod
    def terminate_sagemaker_job(jobname):
        client = boto3.client("sagemaker")
        try:
            client.stop_training_job(TrainingJobName=jobname)
        except Exception as e:
            print(e)

    @staticmethod
    def add_tags(sm_job_name, tags):
        client = boto3.client("sagemaker")
        # TODO create resource arn here
        resource_arn = "arn:aws:sagemaker:us-east-1:072677473360:training-job/" + sm_job_name
        try:
            client.add_tags(ResourceArn=resource_arn, Tags=tags)
        except Exception as e:
            print(e)
