# Standard Library
import asyncio
import logging
import multiprocessing
import time
from functools import lru_cache

# Third Party
import aioboto3
import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    CredentialRetrievalError,
    NoCredentialsError,
    NoRegionError,
    PartialCredentialsError,
    ServiceNotInRegionError,
    UnknownSignatureVersionError,
)

# First Party
from smdebug.core.logger import get_logger
from smdebug.core.utils import get_region, is_s3

logger = get_logger()


def check_notebook():
    # try to see if we are in an iPython environment and import nest_asyncio appropriately
    #
    try:
        get_ipython()
        import nest_asyncio

        nest_asyncio.apply()
    except NameError:
        pass


# Must be created for ANY file read request, whether from S3 or Local
# If you wish to download entire file, leave length as None and start as 0.
# If length is None, start must be 0.
# Full S3 path is required if you wish to download file from s3
# eg s3://ljain-tests/demos/ ....
class ReadObjectRequest:
    def __init__(self, path, start=None, length=None):
        self.is_s3, self.bucket, self.key = is_s3(path)
        if not self.is_s3:
            self.key = path
            self.bucket = None
        if start is None:
            self.start = 0
        else:
            self.start = start
        self.length = length
        self.download_entire_file = start is None


# Only to list files in S3. Accepts strings Bucket, Prefix, Delimiter, and StartAfter
# parameters that serve the same role here as they would in boto3.
class ListRequest:
    def __init__(self, Bucket, Prefix="", Delimiter="", StartAfter=""):
        self.bucket = Bucket
        self.prefix = Prefix
        self.delimiter = Delimiter
        self.start_after = StartAfter


class DeleteRequest:
    def __init__(self, Bucket, Prefix=""):
        self.bucket = Bucket
        self.prefix = Prefix


class S3HandlerAsync:
    # For debug flag, first set PYTHONASYNCIODEBUG=1 in terminal.
    # This provides terminal output revealing details about the AsyncIO calls and timings that may be useful.
    # num_retries: the number of times to retry a download or connection before logging an exception.

    def __init__(self, num_retries=5, debug=False):
        # if you are creating an s3handler object in jupyter, ensure the nest_asyncio is applied
        check_notebook()
        self.loop = asyncio.get_event_loop()
        self.client = aioboto3.client("s3", region_name=get_region())
        self.num_retries = num_retries
        self.logger = get_logger()
        if debug:
            self.loop.set_debug(True)
            self.loop.slow_callback_duration = 4
            logging.basicConfig(level=logging.DEBUG)
            aioboto3.set_stream_logger(name="boto3", level=logging.DEBUG, format_string=None)

    # Accepts a bucket, prefix, delimiter, and start token
    # and returns list of all file names from that bucket that matches the given configuration
    # E.g. bucket='ljain-tests', prefix="rand_1mb_1000", start_after="rand_1mb_1000/demo_96.out.tfevents"
    # Would list all files in the subdirectory 'ljain-tests/rand_1mb_1000' with filenames lexicographically
    # less than demo_96.out.tfevents.
    # If request made is invalid, this returns an empty list and logs the exception in the log.
    #
    async def _list_files(self, bucket, prefix="", delimiter="", start_after=""):
        count = 0
        success = False
        # try num_retries times to establish a connection and download the file
        # if none can be established, log an error and exit
        while count < self.num_retries and not success:
            try:
                paginator = self.client.get_paginator("list_objects_v2")
                page_iterator = paginator.paginate(
                    Bucket=bucket,
                    Prefix=prefix,
                    Delimiter=delimiter,
                    StartAfter=start_after,
                    PaginationConfig={"PageSize": 1000},
                )
                success = True
            except (
                NoCredentialsError,  # No credentials could be found
                PartialCredentialsError,  # Only partial credentials were found.
                CredentialRetrievalError,  # Error attempting to retrieve credentials from a remote source.
                UnknownSignatureVersionError,  # Requested Signature Version is not known.
                ServiceNotInRegionError,  # The service is not available in requested region.
                NoRegionError,  # No region was specified.
                ClientError,  # Covers cases when the client has insufficient permissions
            ) as botocore_error:
                if isinstance(botocore_error, ClientError):
                    if botocore_error.response["Error"]["Code"] != "AccessDenied":
                        self.logger.warning(
                            f"Unable to list files "
                            f"from  {bucket}/{str(prefix)}: {str(botocore_error)}"
                        )
                    else:
                        raise botocore_error
                else:
                    raise botocore_error
            except Exception as e:
                self.logger.warning(str(e))
            count += 1
        if not success:
            self.logger.warning("Unable to list files for " + bucket + " with prefix " + prefix)
            return []
        keys = []
        async for page in page_iterator:
            if delimiter:
                if "CommonPrefixes" in page.keys():
                    for pre in page["CommonPrefixes"]:
                        if "Prefix" in pre.keys():
                            keys += [pre["Prefix"]]
                if "Contents" in page.keys():
                    for obj in page["Contents"]:
                        if "Key" in obj.keys():
                            keys += [obj["Key"]]
            else:
                if "Contents" in page.keys():
                    for obj in page["Contents"]:
                        if "Key" in obj.keys():
                            keys += [obj["Key"]]
        return keys

    # accepts a bucket and key and fetches data from s3 beginning at the offset = start with the provided length
    # If length is None, download the entire file.
    async def _get_object(self, bucket, key, start, length):
        count, body = 0, None
        while count < self.num_retries and body is None:
            try:
                if start is not None:
                    bytes_range = f"bytes={start}-"
                    if length is not None:
                        bytes_range += f"{start + length - 1}"
                    resp = await self.client.get_object(Bucket=bucket, Key=key, Range=bytes_range)
                else:
                    resp = await self.client.get_object(Bucket=bucket, Key=key)
                body = await resp["Body"].read()
            except (
                NoCredentialsError,  # No credentials could be found
                PartialCredentialsError,  # Only partial credentials were found.
                CredentialRetrievalError,  # Error attempting to retrieve credentials from a remote source.
                UnknownSignatureVersionError,  # Requested Signature Version is not known.
                ServiceNotInRegionError,  # The service is not available in requested region.
                NoRegionError,  # No region was specified.
                ClientError,  # Covers cases when the client has insufficient permissions
            ) as botocore_error:
                if isinstance(botocore_error, ClientError):
                    if botocore_error.response["Error"]["Code"] != "AccessDenied":
                        self.logger.warning(
                            f"Unable to read tensor "
                            f"from object {bucket}/{str(key)}: {str(botocore_error)}"
                        )
                        body = None
                    else:
                        raise botocore_error
                else:
                    raise botocore_error
            except Exception as e:
                self.logger.warning(str(e))
                body = None
                msg = "Unable to read tensor from object " + str(bucket) + "/" + str(key)
                if length is not None:
                    msg += " from bytes " + str(start) + "-" + str(start + length - 1)
                self.logger.warning(msg)
            count += 1
        if body is None:
            self.logger.warning("Unable to find file " + str(key))
        return body

    # object_requests: a list of ObjectRequest objects.
    # Returns list of all data fetched.
    async def _get_objects(self, object_requests):
        request_params = []
        for obj in object_requests:
            request_params += [(obj.bucket, obj.key, obj.start, obj.length)]
        data = await asyncio.gather(
            *[
                self._get_object(bucket, key, start, length)
                for bucket, key, start, length in request_params
            ]
        )
        return data

    # list_requests: a list of ListRequest objects
    # Returns list of lists of files fetched
    async def _list_files_from_requests(self, list_requests):
        request_params = []
        for req in list_requests:
            request_params += [(req.bucket, req.prefix, req.delimiter, req.start_after)]
        data = await asyncio.gather(
            *[
                self._list_files(bucket, prefix, delimiter, start_after)
                for bucket, prefix, delimiter, start_after in request_params
            ]
        )
        return data

    # Closes the client
    async def _close_client(self):
        await self.client.close()

    # object_requests: list of ObjectRequest objects.
    # num_async_calls: the number of asynchronous calls to do at once.
    # timer: boolean, allows timing measurement of total operation
    # returns list of all data downloaded.
    def get_objects(self, object_requests, num_async_calls=500, timer=False):
        if type(object_requests) != list:
            raise TypeError("get_objects accepts a list of ObjectRequest objects.")
        if timer:
            start = time.time()
        idx = 0
        data = []
        while idx < len(object_requests):
            task = self.loop.create_task(
                self._get_objects(object_requests[idx : idx + num_async_calls])
            )
            done = self.loop.run_until_complete(task)
            data += done
            idx += num_async_calls
        if timer:
            self.logger.info(
                "total time taken for "
                + str(len(object_requests))
                + " requests: "
                + str(time.time() - start)
            )
        return data

    # accepts a list of ListRequest objects, returns list of lists of files fetched.
    def list_prefixes(self, list_requests: list):
        task = self.loop.create_task(self._list_files_from_requests(list_requests))
        done = self.loop.run_until_complete(task)
        return done

    # Public facing function to close the client.
    def close_client(self):
        task = self.loop.create_task(self._close_client())
        done = self.loop.run_until_complete(task)
        return done

    # Destructor to close client upon deletion
    def __del__(self):
        self.close_client()


class S3Handler:
    NUM_RETRIES = 5

    # A boto3 session is not pickleable, and an object must be pickleable to be accessed within a
    # multiprocessing thread. We get around this by defining a function to create the session - the
    # function is pickleable - and caching the results so it is only called once.
    @staticmethod
    @lru_cache()
    def client():
        return boto3.client("s3", region_name=get_region())

    @staticmethod
    def list_prefixes(list_requests: list):
        if type(list_requests) != list:
            raise TypeError("list_requests accepts a list of ListRequest objects.")
        rval = []
        for lr in list_requests:
            rval.append(S3Handler.list_prefix(lr))
        return rval

    @staticmethod
    def list_prefix(lr):
        count = 0
        while count < S3Handler.NUM_RETRIES:
            try:
                paginator = S3Handler.client().get_paginator("list_objects_v2")
                page_iterator = paginator.paginate(
                    Bucket=lr.bucket,
                    Prefix=lr.prefix,
                    Delimiter=lr.delimiter,
                    StartAfter=lr.start_after,
                    PaginationConfig={"PageSize": 1000},
                )
                keys = []
                for page in page_iterator:
                    if lr.delimiter:
                        if "CommonPrefixes" in page.keys():
                            for pre in page["CommonPrefixes"]:
                                if "Prefix" in pre.keys():
                                    keys += [pre["Prefix"]]
                        if "Contents" in page.keys():
                            for obj in page["Contents"]:
                                if "Key" in obj.keys():
                                    keys += [obj["Key"]]
                    else:
                        if "Contents" in page.keys():
                            for obj in page["Contents"]:
                                if "Key" in obj.keys():
                                    keys += [obj["Key"]]
                return keys
            except (
                NoCredentialsError,  # No credentials could be found
                PartialCredentialsError,
                # Only partial credentials were found.
                CredentialRetrievalError,
                # Error attempting to retrieve credentials from a remote source.
                UnknownSignatureVersionError,
                # Requested Signature Version is not known.
                ServiceNotInRegionError,
                # The service is not available in requested region.
                NoRegionError,  # No region was specified.
                ClientError,
                # Covers cases when the client has insufficient permissions
            ) as botocore_error:
                if isinstance(botocore_error, ClientError):
                    if botocore_error.response["Error"]["Code"] != "AccessDenied":
                        logger.warning(
                            f"Unable to list files "
                            f"from  {lr.bucket}/{str(lr.prefix)}: {str(botocore_error)}"
                        )
                    else:
                        raise botocore_error
                else:
                    raise botocore_error
            except Exception as e:
                logger.warning(str(e))
            count += 1

        # if success, we wouldn't come here
        logger.warning(f"Unable to list files for {lr.bucket} with prefix {lr.prefix}")
        return []

    @staticmethod
    def _make_get_request(object_request):
        if not object_request.download_entire_file:
            bytes_range = f"bytes={object_request.start}-"
            if object_request.length is not None:
                bytes_range += f"{object_request.start + object_request.length - 1}"
            body = (
                S3Handler.client()
                .get_object(
                    Bucket=object_request.bucket, Key=object_request.key, Range=bytes_range
                )["Body"]
                .read()
            )
        else:
            body = (
                S3Handler.client()
                .get_object(Bucket=object_request.bucket, Key=object_request.key)["Body"]
                .read()
            )
        return body

    @staticmethod
    def _log_error(object_request, exception=None, as_warning=True):
        if exception:
            msg = f"Encountered the exception {exception} while reading "
        else:
            msg = f"Failed to read "
        msg += f"s3://{object_request.bucket}/{object_request.key} "
        if not object_request.download_entire_file:
            msg += f"from bytes {object_request.start}"
            if object_request.length is not None:
                msg += f"-{object_request.start + object_request.length - 1}"
        if as_warning:
            msg += ". Will retry now"
            logger.warning(msg)
        else:
            logger.error(msg)

    @staticmethod
    def get_object(object_request):
        count = 0
        while count < S3Handler.NUM_RETRIES:
            try:
                return S3Handler._make_get_request(object_request)
            except (
                NoCredentialsError,  # No credentials could be found
                PartialCredentialsError,  # Only partial credentials were found
                CredentialRetrievalError,  # Error attempting to retrieve credentials
                UnknownSignatureVersionError,  # Requested Signature Version is not known
                ServiceNotInRegionError,  # Service not available in requested region
                NoRegionError,  # No region was specified.
                ClientError,  # Covers cases when the client has insufficient permissions
            ) as botocore_error:
                if (
                    not isinstance(botocore_error, ClientError)
                    or botocore_error.response["Error"]["Code"] == "AccessDenied"
                ):
                    raise botocore_error
                else:
                    S3Handler._log_error(object_request, exception=botocore_error)
            except Exception as e:
                S3Handler._log_error(object_request, exception=e)
            count += 1
            time.sleep(0.1)
        S3Handler._log_error(object_request, as_warning=False)
        return None

    @staticmethod
    def get_objects(object_requests, use_multiprocessing=True):
        if type(object_requests) != list:
            raise TypeError("get_objects accepts a list of ReadObjectRequest objects")
        if use_multiprocessing:
            with multiprocessing.Pool(8 * multiprocessing.cpu_count()) as pool:
                data = pool.map(S3Handler.get_object, object_requests)
        else:
            data = [S3Handler.get_object(object_request) for object_request in object_requests]
        return data

    @staticmethod
    def delete_prefix(path, delete_request=None):
        if path is not None and delete_request is not None:
            raise ValueError("Only one of path or delete_request can be passed")
        elif path is not None:
            on_s3, bucket, prefix = is_s3(path)
            if on_s3 is False:
                raise ValueError("Given path is not an S3 location")
            delete_requests = [DeleteRequest(bucket, prefix)]
            S3Handler.delete_prefixes(delete_requests)
        elif delete_request is not None:
            S3Handler.delete_prefixes([delete_request])

    @staticmethod
    def delete_prefixes(delete_requests):
        if delete_requests is not None and len(delete_requests) == 0:
            return
        for delete_request in delete_requests:
            list_request = ListRequest(Bucket=delete_request.bucket, Prefix=delete_request.prefix)
            keys = S3Handler.list_prefix(list_request)
            if len(keys):
                delete_dict = {}
                delete_dict["Objects"] = []
                for key in keys:
                    delete_dict["Objects"].append({"Key": key})
                S3Handler.client().delete_objects(Bucket=delete_request.bucket, Delete=delete_dict)
