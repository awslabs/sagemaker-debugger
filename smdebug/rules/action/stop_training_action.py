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
        res = {}
        found_job_dict = {}
        next_token = None
        name = self._training_job_prefix
        i = 0
        exception_caught_times = 0
        while i < 50:
            try:
                if next_token is None:
                    res = self._sm_client.list_training_jobs(
                        NameContains=name,
                        SortBy="CreationTime",
                        SortOrder="Descending",
                        StatusEquals="InProgress",
                    )
                else:
                    res = self._sm_client.list_training_jobs(
                        NextToken=next_token,
                        NameContains=name,
                        SortBy="CreationTime",
                        SortOrder="Descending",
                        StatusEquals="InProgress",
                    )
                if "TrainingJobSummaries" in res:
                    jobs = res["TrainingJobSummaries"]
                else:
                    self._logger.info(
                        f"No TrainingJob summaries found: list_training_jobs output is : {res}"
                    )
                    return []
                for job in jobs:
                    tj_status = job["TrainingJobStatus"]
                    tj_name = job["TrainingJobName"]
                    self._logger.info(f"TrainingJob name: {tj_name} , status:{tj_status}")
                    if tj_name is not None and tj_name.startswith(name):
                        found_job_dict[tj_name] = 1
                self._logger.info(f"found_training job {found_job_dict.keys()}")
            except Exception as e:
                self._logger.info(
                    f"Caught exception while getting list_training_job exception is: \n {e}. Attempt:{i}"
                )
                exception_caught_times += 1
                if exception_caught_times > 5:
                    print("Got exception more than 5 times while finding training job. Giving up.")
                    break
            if "NextToken" not in res:
                break
            else:
                next_token = res["NextToken"]
                res = {}
                jobs = {}
                i += 1
            if len(found_job_dict) > 0:
                print(
                    f"Found training jobs matching prefix:{name}. Exiting even if next_token:{next_token} was present."
                )
                break

        return list(found_job_dict.keys())

    def _stop_training_job(self, message):
        if len(self._found_jobs) != 1:
            return
        if message != "":
            message = f"with message {message}"
        self._logger.info(
            f"Invoking StopTrainingJob action on SM jobname {self._found_jobs} {message}"
        )
        try:
            res = self._sm_client.stop_training_job(TrainingJobName=self._found_jobs[0])
            self._logger.info(f"Stop Training job response:{res}")
        except Exception as e:
            self._logger.info(f"Got exception while stopping training job{self._found_jobs[0]}:{e}")

    def invoke(self, message):
        self._stop_training_job(message)
