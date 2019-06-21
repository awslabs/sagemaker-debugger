from ObjectRequest import *
import aioboto3
import asyncio

class ObjectRequest:
    def __init__(self, path, start=0, length=None):
        # if you wish to download entire file, leave length as None
        # and this will download the entire file
        self.path = path # The S3 url: e.g. s3://ljain-tests/tensors/tensor_1.npy
        self.bucket = None
        self.is_s3 = (path[:5] == "s3://")
        if self.is_s3:
            self.bucket = path[5:].split("/")[0]
            self.path = '/'.join(path[5:].split("/")[1:])
        self.start = start
        self.length = length
        self.download_entire_file = (self.start == 0 and self.length is None)

class S3Handler:
    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.client = aioboto3.session.Session().client('s3')

    async def _list_files(self, bucket, substr_match=""):
        paginator = self.client.get_paginator('list_objects')
        page_iterator = paginator.paginate(Bucket=bucket, PaginationConfig={'PageSize': 100})
        keys = []
        async for page in page_iterator:
            for obj in page['Contents']:
                key = obj['Key']
                if substr_match in key:
                    keys += [key]
        return keys

    async def _get_object(self, bucket, key, start, length):
        try:
            if length is not None:
                bytes_range = "bytes=" + str(start) + "-" + str(start + length)
                resp = await self.client.get_object(Bucket=bucket, Key=key, bytes_range=bytes_range)
            else:
                resp = await self.client.get_object(Bucket=bucket, Key=key)
            body = await resp['Body'].read()
        except Exception as e:
            print(e)
            body = None
            print("Unable to find file " + key)
        return body

    async def _get_objects(self, object_requests):
        request_params = []
        for obj in object_requests:
            request_params += [(obj.bucket, obj.path, obj.start, obj.length)]
        data = await asyncio.gather(*[self._get_object(bucket, key, start, length) for bucket, key, start, length in request_params])
        return data

    async def _close_client(self):
        await self.client.close()

    def get_objects(self, object_requests):
        task = self.loop.create_task(self._get_objects(object_requests))
        done = self.loop.run_until_complete(task)
        return done 

    def list_files(self, bucket):
        task = self.loop.create_task(self._list_files(bucket))
        done = self.loop.run_until_complete(task)
        return done 

    def close_client(self):
        task = self.loop.create_task(self._close_client())
        done = self.loop.run_until_complete(task)
        return done

