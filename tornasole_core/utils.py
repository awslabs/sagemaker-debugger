import os
import re
import logging
from botocore.exceptions import ClientError


def flatten(lis):
  """Given a list, possibly nested to any level, return it flattened."""
  new_lis = []
  for item in lis:
      if type(item) == type([]):
          new_lis.extend(flatten(item))
      else:
          new_lis.append(item)
  return new_lis

logger = None

def get_logger(path=os.getcwd()):
  global logger
  if logger is None:
    logger = logging.getLogger("tornasole")
    fh = logging.FileHandler(os.path.join(path, 'tornasole.log'))
    log_level = os.environ.get('TORNASOLE_LOG_LEVEL', default='info')
    log_level = log_level.lower()

    if log_level not in ['info', 'debug', 'warning', 'error', 'critical', 'off']:
      print('Invalid log level for TORNASOLE_LOG_LEVEL')
      log_level = 'info'

    if log_level == 'off':
      logger.disabled = True
    elif log_level == 'critical':
      logger.setLevel(logging.CRITICAL)
    elif log_level == 'error':
      logger.setLevel(logging.ERROR)
    elif log_level == 'warning':
      logger.setLevel(logging.WARNING)
    elif log_level == 'info':
      logger.setLevel(logging.INFO)
    elif log_level == 'debug':
      logger.setLevel(logging.DEBUG)

    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
  # logger.propagate = False
    return logger
  else:
    return logger


def get_immediate_subdirectories(a_dir):
  return [name for name in os.listdir(a_dir)
          if os.path.isdir(os.path.join(a_dir, name))]

TORNASOLE_REDUCTIONS_PREFIX = "tornasole/reductions/"

def get_reduction_tensor_name(tensorname, reduction_name, abs):
    tname = re.sub(r':\d+', '', f'{reduction_name}/{tensorname}')
    if abs:
        tname = 'abs_' + tname
    tname = TORNASOLE_REDUCTIONS_PREFIX + tname
    return tname

def reverse_reduction_tensor_name(reduction_tensor_name):
    rest = reduction_tensor_name.split(TORNASOLE_REDUCTIONS_PREFIX)[1]
    parts = rest.split('/', 1)
    reduction_name = parts[0]
    if 'abs_' in reduction_name:
        abs = True
        reduction_op_name = reduction_name.split('abs_')[1]
    else:
        abs = False
        reduction_op_name = reduction_name
    tensor_name = parts[1]
    return tensor_name, reduction_op_name, abs

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
    from tornasole_core.access_layer.s3handler import S3Handler, ListRequest
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


def match_inc(tname, include):
    for inc in include:
        if re.search(inc, tname):
            return True
    return False