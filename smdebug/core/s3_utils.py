# First Party
from smdebug.core.access_layer.s3handler import ListRequest, S3Handler


# list_info will be a list of ListRequest objects. Returns list of lists of files for each request
def _list_s3_prefixes(list_info):
    files = S3Handler.list_prefixes(list_info)
    if len(files) == 1:
        files = files[0]
    return files


def list_s3_objects(bucket, prefix, start_after_key=None, delimiter=""):
    last_token = None
    if start_after_key is None:
        start_after_key = prefix
    req = ListRequest(Bucket=bucket, Prefix=prefix, StartAfter=start_after_key, Delimiter=delimiter)
    objects = _list_s3_prefixes([req])
    if len(objects) > 0:
        last_token = objects[-1]
    return objects, last_token
