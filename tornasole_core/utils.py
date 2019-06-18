import os
import logging

def flatten(lis):
  """Given a list, possibly nested to any level, return it flattened."""
  new_lis = []
  for item in lis:
      if type(item) == type([]):
          new_lis.extend(flatten(item))
      else:
          new_lis.append(item)
  return new_lis

def get_logger(path):
  logger = logging.getLogger("tornasole_logger")
  fh = logging.FileHandler(os.path.join(path, 'tornasole.log'))
  log_level = os.environ.get('TORNASOLE_LOG_LEVEL', default=5)
  try:
    log_level = int(log_level)
  except ValueError:
    print('TORNASOLE_LOG_LEVEL can only be an integer')
  if log_level == 0:
    logger.disabled = True
  elif log_level == 1:
    logger.setLevel(logging.CRITICAL)
  elif log_level == 2:
    logger.setLevel(logging.ERROR)
  elif log_level == 3:
    logger.setLevel(logging.WARNING)
  elif log_level == 4:
    logger.setLevel(logging.INFO)
  elif log_level == 5:
    logger.setLevel(logging.DEBUG)

  fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
  logger.addHandler(fh)
  # logger.propagate = False
  return logger
