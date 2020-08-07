# Standard Library
import multiprocessing
import sys
import time
from functools import lru_cache

# Third Party
import boto3
from botocore.exceptions import (
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


class ReadObjectRequest:
    """
    If you wish to download entire file, leave length and start as None
    Full S3 path is required if you wish to download file from s3
    eg s3://ljain-tests/demos/ ....
    """

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


class ListRequest:
    """
    Only to list files in S3. Accepts strings Bucket, Prefix, Delimiter, and StartAfter
    parameters that serve the same role here as they would in boto3.
    """

    def __init__(self, Bucket, Prefix="", Delimiter="", StartAfter=""):
        self.bucket = Bucket
        self.prefix = Prefix
        self.delimiter = Delimiter
        self.start_after = StartAfter


class DeleteRequest:
    def __init__(self, Bucket, Prefix=""):
        self.bucket = Bucket
        self.prefix = Prefix


class S3Handler:
    NUM_RETRIES = 5
    GET_OBJECTS_MULTIPROCESSING_THRESHOLD = 100
    MULTIPROCESSING_POOL_SIZE = 2 * multiprocessing.cpu_count()

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
        if (
            use_multiprocessing
            and len(object_requests) >= S3Handler.GET_OBJECTS_MULTIPROCESSING_THRESHOLD
            and sys.platform != "win32"  # Windows Jupyter has trouble with multiprocessing
        ):
            if sys.platform == "darwin":
                # Mac will crash if we use fork(), which is the default until Python 3.8+
                ctx = multiprocessing.get_context("spawn")
            else:
                ctx = multiprocessing.get_context()
            with ctx.Pool(S3Handler.MULTIPROCESSING_POOL_SIZE) as pool:
                data = pool.map(S3Handler.get_object, object_requests)
        else:
            data = [S3Handler.get_object(object_request) for object_request in object_requests]
        return data

    @staticmethod
    def delete_prefix(path=None, delete_request=None):
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
