import re

def is_s3(path):
    m = re.match(r's3://([^/]+)/(.*)', path)
    if not m:
        return (False, None, None)
    return (True, m[1], m[2])