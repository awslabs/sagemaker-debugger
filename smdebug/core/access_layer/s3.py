# Standard Library
import os
import re

# Third Party
import boto3

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

        # check if the bucket exists
        buckets = [bucket["Name"] for bucket in self.s3_client.list_buckets()["Buckets"]]
        if self.bucket_name not in buckets:
            self.s3_client.create_bucket(ACL="private", Bucket=self.bucket_name)

    def _init_data(self):
        if self.binary:
            self.data = bytearray()
        else:
            self.data = ""

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
        key = self.s3.Object(self.bucket_name, self.key_name)
        key.put(Body=self.data)
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
