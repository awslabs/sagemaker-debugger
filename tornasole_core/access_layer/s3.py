import boto
import boto3
import re
from tornasole_core.access_layer.base import TSAccessBase

class TSAccessS3(TSAccessBase):
    def __init__(self, bucket_name, key_name,
                 aws_access_key_id=None, aws_secret_access_key=None,
                 binary=True):
        super().__init__()
        self.bucket_name = bucket_name
        # S3 is not like a Unix file system where multiple slashes are normalized to one
        self.key_name = re.sub('/+', '/', key_name )
        self.binary = binary
        self._init_data()
        self.flushed = False

    def _init_data(self):
        if self.binary:
            self.data = bytearray()
        else:
            self.data = ''

    def open(self,bucket_name,mode):
        raise NotImplementedError

    def write(self, _data):
        self.data += _data

    def close(self):
        if self.flushed:
            return
        s3 = boto3.resource('s3')
        key = s3.Object(self.bucket_name, self.key_name)
        key.put(Body=self.data)
        self._init_data()
        self.flushed = True


    def flush(self):
        pass

    def ingest_all(self):
        s3_client = boto3.client('s3')
        s3_response_object = s3_client.get_object(Bucket=self.bucket_name, Key=self.key_name)
        self._data = s3_response_object['Body'].read()
        self._datalen = len(self._data)
        self._position = 0
    
    def read(self,n):
        assert self._position+n <= self._datalen
        res = self._data[self._position:self._position+n]
        self._position += n
        return res

    def has_data(self):
        return self._position < self._datalen

    def __enter__(self):
        return self

    def __exit(self, exc_type, exc_value, traceback):
        self.close()
