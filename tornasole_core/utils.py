import os
import re
import logging


DEFAULT_PREFIX_FOR_S3_TRIAL_NAME = "__trial__"

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

def get_reduction_tensor_name(tensorname, reduction_name, abs):
    tname = re.sub(r':\d+', '', f'{reduction_name}/{tensorname}')
    if abs:
        tname = 'abs_' + tname
    tname = "tornasole/reductions/" + tname
    return tname

def is_s3(path):
    m = re.match(r's3://([^/]+)/(.*)', path)
    if not m:
        return (False, None, None)
    return (True, m[1], m[2])

def check_dir_exists(path):
    from tornasole_core.access_layer.s3handler import S3Handler, ListRequest
    s3, bucket_name, key_name = is_s3(path)
    if s3:
        s3_handler = S3Handler()
        request = ListRequest(bucket_name, key_name)
        folder = s3_handler.list_prefixes([request])[0]
        if len(folder) > 0:
            raise RuntimeError('The path:{} already exists in s3'.format(path))
    elif os.path.exists(path):
        raise RuntimeError('The path:{} already exists in local disk'.format(path))
