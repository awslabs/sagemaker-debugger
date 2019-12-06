# Standard Library
import os


def is_sagemaker_job():
    """
    If this variable is defined we are assuming that this is
    a Sagemaker job. This is guaranteed to be defined
    for all Sagemaker jobs.
    https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-environment-variables
    :return: True or False
    """
    return "TRAINING_JOB_NAME" in os.environ
