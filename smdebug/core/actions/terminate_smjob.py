# First Party
from smdebug.core.sagemaker_utils import SageMakerUtils

# Local
from .action_base import Action


class TerminateSagemakerJob(Action):
    def __init__(self, sm_job_name):
        super().__init__()
        self.job_name = sm_job_name

    def run(self, rule_name, **kwargs):
        try:
            # todo fix hardcoding of arn inside this function
            SageMakerUtils.terminate_sagemaker_job(self.job_name)
            # tags = [{'Key':"TerminatedBy", 'Value': rule_name} ,
            #         {'Key':'TerminationTime', 'Value': str(time.time())} ]
            # SageMakerUtils.add_tags(self.job_name, tags)
        except Exception as e:
            self.logger.warning(
                "Caught exception when running TerminateSagemakerJob "
                "action for smjob:{} Exception:{}".format(self.sm_job_name, e)
            )
