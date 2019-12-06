# Standard Library
import os

# Third Party
from botocore.exceptions import ClientError

# First Party
from smdebug.core.access_layer.s3handler import DeleteRequest, ListRequest, S3Handler
from smdebug.core.logger import get_logger
from smdebug.core.sagemaker_utils import is_sagemaker_job
from smdebug.core.utils import is_s3

# Local
from .file import TSAccessFile
from .s3 import TSAccessS3

END_OF_JOB_FILENAME = "training_job_end.ts"
logger = get_logger()


def training_has_ended(trial_prefix):
    # Emit the end of training file only if the job is not running under SageMaker.
    if is_sagemaker_job():
        logger.info(
            f"The end of training job file will not be written for jobs running under SageMaker."
        )
        return
    try:
        check_dir_exists(trial_prefix)
        # if path does not exist, then we don't need to write a file
    except RuntimeError:
        # dir exists
        pass
    file_path = os.path.join(trial_prefix, END_OF_JOB_FILENAME)
    s3, bucket_name, key_name = is_s3(file_path)
    if s3:
        writer = TSAccessS3(bucket_name, key_name, binary=False)
    else:
        writer = TSAccessFile(file_path, "a+")
    writer.flush()
    try:
        writer.close()
    except OSError:
        """
        In the case of distributed training in local mode,
        another worker may have already moved the END_OF_JOB file
        from the /tmp directory.
        """


def has_training_ended(trial_prefix):
    file_path = os.path.join(trial_prefix, END_OF_JOB_FILENAME)
    s3, bucket_name, key_name = is_s3(file_path)
    if s3:
        try:
            request = ListRequest(bucket_name, key_name)
            file_available = S3Handler.list_prefixes([request])[0]
            if len(file_available) > 0:
                return True
            else:
                return False
        except ClientError as ex:
            status_code = ex.response["ResponseMetadata"]["HTTPStatusCode"]
            logger.info(f"Client error occurred : {ex}")
            if status_code.startswith("4"):
                raise ex
            else:
                return False
    else:
        return os.path.exists(file_path)


def delete_s3_prefixes(bucket, keys):
    if not isinstance(keys, list):
        keys = [keys]
    delreqs = []
    for key in keys:
        delreqs.append(DeleteRequest(bucket, key))
    S3Handler.delete_prefixes(delreqs)


def check_dir_exists(path):
    from smdebug.core.access_layer.s3handler import S3Handler, ListRequest

    s3, bucket_name, key_name = is_s3(path)
    if s3:
        try:
            request = ListRequest(bucket_name, key_name)
            folder = S3Handler.list_prefixes([request])[0]
            if len(folder) > 0 and has_training_ended(folder[-1]):
                raise RuntimeError(
                    "The path:{} already exists on s3. "
                    "Please provide a directory path that does "
                    "not already exist.".format(path)
                )
        except ClientError as ex:
            if ex.response["Error"]["Code"] == "NoSuchBucket":
                # then we do not need to raise any error
                pass
            else:
                # do not know the error
                raise ex
    elif os.path.exists(path) and has_training_ended(path):
        raise RuntimeError(
            "The path:{} already exists on local disk. "
            "Please provide a directory path that does "
            "not already exist".format(path)
        )
