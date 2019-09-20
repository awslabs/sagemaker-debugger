import os
import re
import bisect
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Dict, List

def flatten(lis):
  """Given a list, possibly nested to any level, return it flattened."""
  new_lis = []
  for item in lis:
      if type(item) == type([]):
          new_lis.extend(flatten(item))
      else:
          new_lis.append(item)
  return new_lis

def split(comma_separated_string: str) -> List[str]:
    """Split "foo, bar,b az" into ["foo","bar","b az".]"""
    return [x.strip() for x in comma_separated_string.split(",")]

def merge_two_dicts(x, y) -> Dict:
    """If x and y have the same key, then y's value takes precedence.

    For example, merging
        x = {'a': 1, 'b': 2},
        y = {'b': 3, 'c': 4}
        yields
        z = {'a': 1, 'b': 3, 'c': 4}.
    """
    return {**x, **y}

def get_immediate_subdirectories(a_dir):
  return [name for name in os.listdir(a_dir)
          if os.path.isdir(os.path.join(a_dir, name))]


def is_s3(path):
    if path.startswith('s3://'):
        try:
            parts = path[5:].split('/', 1)
            return True, parts[0], parts[1]
        except IndexError:
            return True, path[5:], ''
    else:
        return False, None, None


def check_dir_exists(path):
    from tornasole.core.access_layer.s3handler import S3Handler, ListRequest
    s3, bucket_name, key_name = is_s3(path)
    if s3:
        try:
            s3_handler = S3Handler()
            request = ListRequest(bucket_name, key_name)
            folder = s3_handler.list_prefixes([request])[0]
            if len(folder) > 0:
                raise RuntimeError('The path:{} already exists on s3. '
                                   'Please provide a directory path that does '
                                   'not already exist.'.format(path))
        except ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchBucket':
                # then we do not need to raise any error
                pass
            else:
                # do not know the error
                raise ex
    elif os.path.exists(path):
        raise RuntimeError('The path:{} already exists on local disk. '
                           'Please provide a directory path that does '
                           'not already exist'.format(path))


def list_files_in_directory(directory):
    files = []
    for root, dir_name, filename in os.walk(directory):
        for f in filename:
            files.append(os.path.join(root, f))
    return files


def match_inc(tname, include):
    for inc in include:
        if re.search(inc, tname):
            return True
    return False


def index(sorted_list, elem):
    i = bisect.bisect_left(sorted_list, elem)
    if i != len(sorted_list) and sorted_list[i] == elem:
        return i
    raise ValueError


def get_region():
    # returns None if region is not set
    region_name = os.environ.get('AWS_REGION')
    if region_name is not None and region_name.strip() == '':
        region_name = None
    return region_name


def step_in_range(range_steps, step):
    if range_steps[0] is not None:
        begin = int(step) >= int(range_steps[0])
    else:
        begin = True
    if range_steps[1] is not None:
        end = int(step) < int(range_steps[1])
    else:
        end = True
    return begin and end


def get_relative_event_file_path(path):
    p = Path(path)
    path_parts = p.parts
    assert path_parts[-3] == "events"
    return os.path.join(*path_parts[-3:])
