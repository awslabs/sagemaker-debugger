import os
import re
import logging
import bisect
from botocore.exceptions import ClientError
import uuid
import sys
import socket


def flatten(lis):
  """Given a list, possibly nested to any level, return it flattened."""
  new_lis = []
  for item in lis:
      if type(item) == type([]):
          new_lis.extend(flatten(item))
      else:
          new_lis.append(item)
  return new_lis

_logger_initialized = False


class MaxLevelFilter(logging.Filter):
    '''Filters (lets through) all messages with level < LEVEL'''
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        # "<" instead of "<=": since logger.setLevel is inclusive, this should be exclusive
        return record.levelno < self.level


def _get_log_level():
    default = 'info'
    log_level = os.environ.get('TORNASOLE_LOG_LEVEL', default=default)
    log_level = log_level.lower()
    allowed_levels = ['info', 'debug', 'warning', 'error', 'critical', 'off']
    if log_level not in allowed_levels:
        log_level = default

    level = None
    if log_level is None or log_level == 'off':
        level = None
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
    return level


def get_logger(name='tornasole'):
    global _logger_initialized
    if not _logger_initialized:
        worker_pid = socket.gethostname() + ':' + str(os.getpid())
        log_context = os.environ.get('TORNASOLE_LOG_CONTEXT', default=worker_pid)
        level = _get_log_level()
        logger = logging.getLogger(name)

        logger.handlers = []
        log_formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d ' + log_context +
                                          ' %(levelname)s %(filename)s:%(lineno)d] %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_formatter)

        if os.environ.get('TORNASOLE_LOG_ALL_TO_STDOUT', default='TRUE').lower() == 'false':
            stderr_handler = logging.StreamHandler(sys.stderr)
            min_level = logging.DEBUG
            # lets through all levels less than ERROR
            stdout_handler.addFilter(MaxLevelFilter(logging.ERROR))
            stdout_handler.setLevel(min_level)

            stderr_handler.setLevel(max(min_level, logging.ERROR))
            stderr_handler.setFormatter(log_formatter)
            logger.addHandler(stderr_handler)

        logger.addHandler(stdout_handler)

        # TORNASOLE_LOG_PATH is the full path to log file
        # by default, log is only written to stdout&stderr
        # if this is set, it is written to file
        path = os.environ.get('TORNASOLE_LOG_PATH', default=None)
        if path is not None:
            fh = logging.FileHandler(path)
            fh.setFormatter(log_formatter)
            logger.addHandler(fh)

        if level:
            logger.setLevel(level)
        else:
            logger.disabled = True
        logger.propagate = False
        _logger_initialized = True
    return logging.getLogger(name)


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
