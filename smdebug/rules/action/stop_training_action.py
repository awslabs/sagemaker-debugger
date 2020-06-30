# Standard Library
import os

# Third Party
import boto3

# First Party
from smdebug.core.logger import get_logger


class StopTrainingAction:
    def __init__(self, rule_name, training_job_prefix):
        self._training_job_prefix = training_job_prefix
        env_region_name = os.getenv("AWS_REGION", "us-east-1")
        self._logger = get_logger()
        self._logger.info(
            f"StopTrainingAction created with training_job_prefix:{training_job_prefix} and region:{env_region_name}"
        )
        self._sm_client = boto3.client("sagemaker", region_name=env_region_name)
        self._rule_name = rule_name
        self._found_jobs = self._get_sm_tj_jobs_with_prefix()

    def _get_sm_tj_jobs_with_prefix(self):
        found_jobs = []
        try:
            jobs = self._sm_client.list_training_jobs()
            if "TrainingJobSummaries" in jobs:
                jobs = jobs["TrainingJobSummaries"]
            else:
                self._logger.info(
                    f"No TrainingJob summaries found: list_training_jobs output is : {jobs}"
                )
                return
            for job in jobs:
                self._logger.info(
                    f"TrainingJob name: {job['TrainingJobName']} , status:{job['TrainingJobStatus']}"
                )
                if job["TrainingJobName"] is not None and job["TrainingJobName"].startswith(
                    self._training_job_prefix
                ):
                    found_jobs.append(job["TrainingJobName"])
            self._logger.info(f"found_training job {found_jobs}")
        except Exception as e:
            self._logger.info(
                f"Caught exception while getting list_training_job exception is: \n {e}"
            )
        return found_jobs

    def _stop_training_job(self):
        if len(self._found_jobs) != 1:
            return
        self._logger.info(f"Invoking StopTrainingJob action on SM jobname:{self._found_jobs}")
        try:
            res = self._sm_client.stop_training_job(TrainingJobName=self._found_jobs[0])
            self._logger.info(f"Stop Training job response:{res}")
        except Exception as e:
            self._logger.info(f"Got exception while stopping training job{self._found_jobs[0]}:{e}")

    def invoke(self, message=None):
        self._stop_training_job()
