import os
from botocore.exceptions import ClientError
from .file import TSAccessFile
from .s3 import TSAccessS3
from tornasole.core.utils import is_s3, get_region
from tornasole.core.logger import get_logger
from tornasole.core.access_layer.s3handler import S3Handler, ListRequest
import asyncio
import aioboto3

END_OF_JOB_FILENAME = "END_OF_JOB.ts"
logger = get_logger()


def training_has_ended(trial_prefix):
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
        pass


def has_training_ended(trial_prefix):
    file_path = os.path.join(trial_prefix, END_OF_JOB_FILENAME)
    s3, bucket_name, key_name = is_s3(file_path)
    if s3:
        try:
            s3_handler = S3Handler()
            request = ListRequest(bucket_name, key_name)
            file_available = s3_handler.list_prefixes([request])[0]
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
    s3_handler = S3Handler()
    if not isinstance(keys, list):
        keys = [keys]
    list_prefixes = s3_handler.list_prefixes(
        [ListRequest(Bucket=bucket, Prefix=key) for key in keys]
    )
    prefixes = [item for sublist in list_prefixes for item in sublist]
    loop = asyncio.get_event_loop()

    async def del_folder(bucket, keys):
        loop = asyncio.get_event_loop()
        client = aioboto3.client("s3", loop=loop, region_name=get_region())
        await asyncio.gather(*[client.delete_object(Bucket=bucket, Key=key) for key in keys])
        await client.close()

    task = loop.create_task(del_folder(bucket, prefixes))
    loop.run_until_complete(task)


def check_dir_exists(path):
    from tornasole.core.access_layer.s3handler import S3Handler, ListRequest

    s3, bucket_name, key_name = is_s3(path)
    if s3:
        try:
            s3_handler = S3Handler()
            request = ListRequest(bucket_name, key_name)
            folder = s3_handler.list_prefixes([request])[0]
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
