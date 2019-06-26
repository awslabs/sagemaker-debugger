from .base import TSAccessBase
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class TSAccessFile(TSAccessBase):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        ensure_dir(path)
        self.open(path, mode)

    def open(self, path, mode):
        self._accessor = open(path, mode)

    def write(self, _str):
        start = self._accessor.tell()
        self._accessor.write(_str)
        length = len(_str)
        return [start, length]

    def flush(self):
        self._accessor.flush()

    def close(self):
        self._accessor.close()

    def ingest_all(self):
        self._data = self._accessor.read()
        self._datalen = len(self._data)
        self._position = 0


    def read(self, n):
        assert self._position + n <= self._datalen
        res = self._data[self._position:self._position + n]
        self._position += n
        return res

    def has_data(self):
        return self._position < self._datalen
