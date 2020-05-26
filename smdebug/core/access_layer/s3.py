# Standard Library
import io
import os
import re
import tempfile

# Third Party
import boto3
import botocore
from boto3.s3.transfer import TransferConfig

# First Party
from smdebug.core.access_layer.base import TSAccessBase
from smdebug.core.logger import get_logger
from smdebug.core.utils import get_region


class TSAccessS3(TSAccessBase):
    def __init__(
        self, bucket_name, key_name, aws_access_key_id=None, aws_secret_access_key=None, binary=True
    ):
        super().__init__()
        self.bucket_name = bucket_name
        # S3 is not like a Unix file system where multiple slashes are normalized to one
        self.key_name = re.sub("/+", "/", key_name)
        self.binary = binary
        self._init_data()
        self.flushed = False
        self.logger = get_logger()

        self.current_len = 0
        self.s3 = boto3.resource("s3", region_name=get_region())
        self.s3_client = boto3.client("s3", region_name=get_region())

        # Set the desired multipart threshold value (5GB)
        MB = 1024 ** 2
        self.transfer_config = TransferConfig(multipart_threshold=5 * MB)

        # Create bucket if does not exist
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except botocore.exceptions.ClientError:
            self.s3_client.create_bucket(ACL="private", Bucket=self.bucket_name)

    def _init_data(self):
        if self.binary:
            self.data = bytearray()
        else:
            self.data = ""

    def open(self, bucket_name, mode):
        raise NotImplementedError

    def write(self, _data):
        start = len(self.data)

        self.data += _data
        length = len(_data)
        return [start, length]

    def close(self):
        if self.flushed:
            return
        if self.binary:
            self.logger.debug(
                f"Sagemaker-Debugger: Writing binary data to s3://{os.path.join(self.bucket_name, self.key_name)}"
            )
            self.s3_client.upload_fileobj(
                io.BytesIO(self.data), self.bucket_name, self.key_name, Config=self.transfer_config
            )
        else:
            f = tempfile.NamedTemporaryFile(mode="w+")
            self.logger.debug(
                f"Sagemaker-Debugger: Writing string data to s3://{os.path.join(self.bucket_name, self.key_name)}"
            )

            f.write(self.data)
            f.flush()
            self.s3_client.upload_file(
                f.name, self.bucket_name, self.key_name, Config=self.transfer_config
            )

        self.logger.debug(
            f"Sagemaker-Debugger: Wrote {len(self.data)} bytes to file "
            f"s3://{os.path.join(self.bucket_name, self.key_name)}"
        )
        self._init_data()
        self.flushed = True

    def flush(self):
        pass

    def ingest_all(self):
        s3_response_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.key_name)
        self._data = s3_response_object["Body"].read()
        self._datalen = len(self._data)
        self._position = 0

    def read(self, n):
        assert self._position + n <= self._datalen
        res = self._data[self._position : self._position + n]
        self._position += n
        return res

    def has_data(self):
        return self._position < self._datalen

    def __enter__(self):
        return self

    def __exit(self, exc_type, exc_value, traceback):
        self.close()
