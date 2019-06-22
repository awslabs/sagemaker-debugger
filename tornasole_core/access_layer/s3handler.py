import aioboto3
import asyncio
from tornasole_core.utils import is_s3, get_logger
import logging
import time
import numpy as np

class ReadObjectRequest:
    def __init__(self, path, start=0, length=None):
        # if you wish to download entire file, leave length as None
        # and this will download the entire file
        self.is_s3, self.bucket, self.path = is_s3(path)
        if not self.is_s3:
            self.path = path
            self.bucket = None
        assert start >= 0
        self.start = start
        self.length = length
        self.download_entire_file = (self.start == 0 and self.length is None)


class ListPrefixRequest:
    def __init__(self, bucket, prefix=""):
        # if you wish to download entire file, leave length as None
        # and this will download the entire file
        self.bucket = bucket
        self.prefix = prefix

class S3Handler:
    def __init__(self, num_retries=5, region_name='us-west-1', debug=False):
        self.loop = asyncio.get_event_loop()
        self.client = aioboto3.client('s3', region_name=region_name, loop=self.loop)
        self.num_retries = num_retries
        self.logger = get_logger()
        # Debugging things that might be useful in case this ever starts failing. Set PYTHONASYNCIODEBUG=1 in terminal.
        if debug:
            self.loop.set_debug(True)
            self.loop.slow_callback_duration = 4
            logging.basicConfig(level=logging.DEBUG)
            aioboto3.set_stream_logger(name='boto3', level=logging.DEBUG, format_string=None)

    async def _list_files(self, bucket, prefix=""):
        count = 0
        success = False
        # try num_retries time to establish a connection and download the file; if none can be established, log an error and exit
        while count < self.num_retries and not success:
            try:
                paginator = self.client.get_paginator('list_objects')
                page_iterator = paginator.paginate(Bucket=bucket, PaginationConfig={'PageSize': 1000})
                success = True
            except Exception as e:
                self.logger.info(str(e))
            count += 1
        if not success:
            self.logger.info("Unable to list files for " + bucket + " with prefix " + prefix)
            return []
        keys = []
        async for page in page_iterator:
            for obj in page['Contents']:
                key = obj['Key']
                if key.startswith(prefix):
                    keys += [key]
        return keys

    async def _get_object(self, bucket, key, start, length):
        count, body = 0, None
        while count < self.num_retries and body is None:
            try:
                if length is not None:
                    bytes_range = "bytes=" + str(start) + "-" + str(start + length)
                    resp = await self.client.get_object(Bucket=bucket, Key=key, bytes_range=bytes_range)
                else:
                    resp = await self.client.get_object(Bucket=bucket, Key=key)
                body = await resp['Body'].read()
            except Exception as e:
                self.logger.info(str(e))
                body = None
            count += 1
        if body is None:
            print("Unable to find file " + key)
        return body

    async def _get_objects(self, object_requests):
        request_params = []
        for obj in object_requests:
            request_params += [(obj.bucket, obj.path, obj.start, obj.length)]
        data = await asyncio.gather(*[self._get_object(bucket, key, start, length) for bucket, key, start, length in request_params])
        return data

    async def _list_files_from_requests(self, list_requests):
        request_params = []
        for req in list_requests:
            request_params += [(req.bucket, req.prefix)]
        data = await asyncio.gather(*[self._list_files(bucket, prefix) for bucket, prefix in request_params])
        return data

    async def _close_client(self):
        await self.client.close()

    def get_objects(self, object_requests, num_async_calls = 500, timer=False):
        if timer:
            start = time.time()
        idx = 0
        data = []
        while idx < len(object_requests):
            task = self.loop.create_task(self._get_objects(object_requests[idx:idx + num_async_calls]))
            done = self.loop.run_until_complete(task)
            data += done
            idx += num_async_calls
        if timer:
            print("total time taken for " + str(len(object_requests)) + " requests: " + str(time.time() - start))
        return done 

    def list_prefixes(self, list_requests):
        task = self.loop.create_task(self._list_files_from_requests(list_requests))
        done = self.loop.run_until_complete(task)
        return done 

    def close_client(self):
        task = self.loop.create_task(self._close_client())
        done = self.loop.run_until_complete(task)
        return done

