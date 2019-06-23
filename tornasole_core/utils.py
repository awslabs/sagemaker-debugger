import os
import re
import logging
import struct
from tornasole_core.tfrecord._crc32c import *
from tornasole_core.tfevent.event_pb2 import Event
from tornasole_core.tfevent.event_file_reader import get_tensor_data
from tornasole_core.tfevent.util import make_tensor_proto
from tornasole_core.tfevent.summary_pb2 import Summary, SummaryMetadata

def flatten(lis):
  """Given a list, possibly nested to any level, return it flattened."""
  new_lis = []
  for item in lis:
      if type(item) == type([]):
          new_lis.extend(flatten(item))
      else:
          new_lis.append(item)
  return new_lis

def get_logger(path=os.getcwd()):
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

def read_record(data):
    payload = None
    strlen_bytes = data[:8]
    data = data[8:]
    # will give you payload for the record, which is essentially the event.
    strlen = struct.unpack('Q', strlen_bytes)[0]
    saved_len_crc = struct.unpack('I', data[:4])[0]
    data = data[4:]
    payload = data[:strlen]
    data = data[strlen:]
    saved_payload_crc = struct.unpack('I', data[:4])[0]
    return payload

def read_tensor_from_record(data):
    event_str = read_record(data)
    event = Event()
    event.ParseFromString(event_str)
    assert event.HasField('summary')
    summ = event.summary
    tensors = []
    for v in summ.value:    
        tensor_name = v.tag
        tensor_data = get_tensor_data(v.tensor)
        tensors += [tensor_data]
    return tensors
