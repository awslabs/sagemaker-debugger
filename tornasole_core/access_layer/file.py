from .base import TSAccessBase
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class TSAccessFile(TSAccessBase):
  def __init__(self, path, mode):
    super().__init__()
    self.path = path
    self.mode = mode
    ensure_dir(path)
    self.open(path, mode)

  def open(self, path, mode):
    self._accessor = open(path, mode)

  def write(self, _str):
    self.logger.debug("writing", len(_str))
    self._accessor.write(_str)

  def flush(self):
    self._accessor.flush()

  def close(self):
    self._accessor.close()

  def ingest_all(self):
    self._data = self._accessor.read()
    self._datalen = len(self._data)
    self._position = 0
    # self.logger.debug("Ingesting All %d" % self._datalen)

  def read(self, n):
    # self.logger.debug(f'Pos={self._position}, N={n}, DataLen={self._datalen}')
    assert self._position + n <= self._datalen
    res = self._data[self._position:self._position + n]
    self._position += n
    return res

  def has_data(self):
    return self._position < self._datalen

