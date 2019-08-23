import os
from botocore.exceptions import ClientError
from .file import TSAccessFile
from .s3 import TSAccessS3
from tornasole.core.utils import is_s3, get_logger, check_dir_exists
from tornasole.core.access_layer.s3handler import S3Handler, ListRequest
import asyncio
import aioboto3

END_OF_JOB_FILENAME = "END_OF_JOB.ts"
logger = get_logger()


def training_has_ended(trial_prefix):
    try:
        check_dir_exists(trial_prefix)
        # if path does not exist, then we don't need to write a file
        return
    except RuntimeError:
        # dir exists
        pass
    file_path=os.path.join(trial_prefix, END_OF_JOB_FILENAME)
    s3, bucket_name, key_name = is_s3(file_path)
    if s3:
        writer = TSAccessS3(bucket_name, key_name, binary=False)
    else:
        writer = TSAccessFile(file_path, 'a+')
    writer.write("end of training job")
    writer.flush()
    writer.close()


def has_training_ended(trial_prefix):
    file_path=os.path.join(trial_prefix, END_OF_JOB_FILENAME)
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
            status_code = ex.response['ResponseMetadata']['HTTPStatusCode']
            logger.info(f'Client error occurred : {ex}')
            if status_code.startswith('4'):
                raise ex
            else:
                return False
    else:
        return os.path.exists(file_path)


def delete_s3_prefixes(bucket, keys):
    s3_handler = S3Handler()
    if not isinstance(keys, list):
        keys = [keys]
    list_prefixes = s3_handler.list_prefixes([ListRequest(Bucket=bucket,
                                                 Prefix=key) for key in keys])
    prefixes = [item for sublist in list_prefixes for item in sublist]
    loop = asyncio.get_event_loop()

    async def del_folder(bucket, keys):
        loop = asyncio.get_event_loop()
        client = aioboto3.client('s3', loop=loop)
        await asyncio.gather(*[client.delete_object(Bucket=bucket, Key=key) for key in keys])
        await client.close()

    task = loop.create_task(del_folder(bucket, prefixes))
    loop.run_until_complete(task)
