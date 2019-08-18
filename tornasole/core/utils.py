import os
import re
import logging
import bisect
from botocore.exceptions import ClientError
import uuid
import sys

def flatten(lis):
  """Given a list, possibly nested to any level, return it flattened."""
  new_lis = []
  for item in lis:
      if type(item) == type([]):
          new_lis.extend(flatten(item))
      else:
          new_lis.append(item)
  return new_lis

_logger_level_set = False
_logger_filter_set = False

guid = str(uuid.uuid4())
log_context = os.environ.get('TORNASOLE_LOG_CONTEXT', default=guid)

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)

def get_logger():
    global _logger_level_set
    global _logger_filter_set
    name = 'tornasole'
    if not _logger_level_set:
        log_level = os.environ.get('TORNASOLE_LOG_LEVEL', default='info')
        # TORNASOLE_LOG_PATH is the full path to log file, by default, log is generated in current dir
        fh = logging.FileHandler(os.environ.get('TORNASOLE_LOG_PATH', default='./tornasole.log'))
        log_level = log_level.lower()
        if log_level not in ['info', 'debug', 'warning', 'error', 'critical', 'off']:
            print('Invalid log level for TORNASOLE_LOG_LEVEL')
            log_level = 'info'
        level = None
        if log_level == 'off':
            print('logging is off')
        else:
            if log_level == 'critical':
                level = logging.CRITICAL
            elif log_level == 'error':
                level = logging.ERROR
            elif log_level == 'warning':
                level = logging.WARNING
            elif log_level == 'info':
                level = logging.INFO
            elif log_level == 'debug':
                level = logging.DEBUG
        _logger_level_set = True
    if not _logger_filter_set:
        log_filter = os.environ.get('TORNASOLE_LOG_FILTER', default=None)
        if log_filter is not None:
            log_filter = log_filter.lower()
        if log_filter not in ['info', 'debug', 'warning', 'error', 'critical', None]:
            print('Invalid log filter for TORNASOLE_LOG_FILTER')
            log_filter = None
        filter = None
        # by default, no filter
        if log_filter is None:
            print('logging filter is off')
        else:
            if log_filter == 'critical':
                filter = logging.CRITICAL
            elif log_filter == 'error':
                filter = logging.ERROR
            elif log_filter == 'warning':
                filter = logging.WARNING
            elif log_filter == 'info':
                filter = logging.INFO
            elif log_filter == 'debug':
                filter = logging.DEBUG
        _logger_filter_set = True
    # create a logger
    logger = logging.getLogger()
    if not len(logger.handlers):
        log_formatter = logging.Formatter('%(asctime)s ' + log_context + ' %(levelname)s %(filename)s(%(lineno)d) %(message)s')
        stdout_handler  = logging.StreamHandler(sys.stdout)
        stderr_handler  = logging.StreamHandler(sys.stderr)
        # adding filter to StreamHandler
        if log_filter is not None:
            f1 = SingleLevelFilter(filter, False)
            f2 = SingleLevelFilter(filter, True)
            # print level == log_filter messages to stderr
            stderr_handler.addFilter(f1)
            # print level != log_filter messages to stdout
            stdout_handler.addFilter(f2)
        # adding filter to FileHandler
        # fh.addFilter(f2)
        stdout_handler.setFormatter(log_formatter)
        stderr_handler.setFormatter(log_formatter)
        fh.setFormatter(log_formatter)
        if level:
            logger.setLevel(level)
        else:
            logger.disabled = True
        logger.addHandler(fh)
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)
    return logger

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
