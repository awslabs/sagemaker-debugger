import os
from botocore.exceptions import ClientError
from .file import TSAccessFile
from .s3 import TSAccessS3
from tornasole_core.utils import is_s3, get_logger
from tornasole_core.access_layer.s3handler import S3Handler, ListRequest

END_OF_JOB_FILENAME = "END_OF_JOB.ts"
logger = get_logger()
def training_has_ended(trial_prefix):
    file_path=os.path.join(trial_prefix, END_OF_JOB_FILENAME)
    s3, bucket_name, key_name = is_s3(file_path)
    writer = None
    if s3:
        writer = TSAccessS3(bucket_name, key_name, binary=False)
    else:
        writer = TSAccessFile(file_path, 'a+')
    writer.write("end of training job")
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