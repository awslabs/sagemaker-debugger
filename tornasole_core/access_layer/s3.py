import boto
import boto3
from tornasole_core.access_layer.base import TSAccessBase

class TSAccessS3(TSAccessBase):
    def __init__(self, bucket_name, key_name, aws_access_key_id=None, aws_secret_access_key=None):
        self.bucket_name = bucket_name
        self.key_name = key_name
        self.s3_connection = boto.connect_s3(aws_access_key_id=aws_access_key_id, 
                                             aws_secret_access_key=aws_secret_access_key)
        self.data = bytearray()
        self.flushed = False

    def open(self,bucket_name,mode):
        raise NotImplementedError

    def write(self, _data):
        #print( "Adding data:", len(_data))
        self.data += _data
        #print( "Current buffer size:", len(self.data))

    def close(self):
        if self.flushed:
            return
        
        #print(f'Writing to {self.key_name}, bytes={len(self.data)}')
        if False:
            self.bucket = self.s3_connection.get_bucket(self.bucket_name)
            self.key = boto.s3.key.Key(self.bucket, self.key_name) 
            self.key.set_contents_from_string(self.data)
        else:
            s3 = boto3.resource('s3')
            key = s3.Object(self.bucket_name, self.key_name)
            key.put(Body=self.data)

        self.data = bytearray()
        self.flushed = True
        pass

    def flush(self):
        pass

    def ingest_all(self):
        s3_client = boto3.client('s3')
        s3_response_object = s3_client.get_object(Bucket=self.bucket_name, Key=self.key_name)
        self._data = s3_response_object['Body'].read()
        self._datalen = len(self._data)
        self._position = 0
        #print( "Ingesting All", self._datalen)
    
    def read(self,n):
        #print( f'Pos={self._position}, N={n}, DataLen={self._datalen}')
        assert self._position+n <= self._datalen
        res = self._data[self._position:self._position+n]
        self._position += n
        return res

    def has_data(self):
        return self._position < self._datalen
