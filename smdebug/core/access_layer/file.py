# Standard Library
import os
import shutil

# First Party
from smdebug.core.logger import get_logger
from smdebug.core.sagemaker_utils import is_sagemaker_job

# Local
from .base import TSAccessBase

NON_SAGEMAKER_TEMP_PATH_PREFIX = "/tmp"
SAGEMAKER_TEMP_PATH_SUFFIX = ".tmp"


def ensure_dir(file_path, is_file=True):
    if is_file:
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_temp_path(file_path):
    directory = os.path.dirname(file_path)
    if is_sagemaker_job():
        temp_path = file_path + SAGEMAKER_TEMP_PATH_SUFFIX
    else:
        if len(file_path) > 0 and file_path[0] == "/":
            file_path = file_path[1:]
        temp_path = os.path.join(NON_SAGEMAKER_TEMP_PATH_PREFIX, file_path)
    return temp_path


WRITE_MODES = ["w", "w+", "wb", "wb+", "a", "a+", "ab", "ab+"]


class TSAccessFile(TSAccessBase):
    def __init__(self, path, mode):
        super().__init__()
        self.path = path
        self.mode = mode
        self.logger = get_logger()
        ensure_dir(path)
        if mode in WRITE_MODES:
            self.temp_path = get_temp_path(self.path)
            ensure_dir(self.temp_path)
            self.open(self.temp_path, mode)
        else:
            self.open(self.path, mode)

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
        """Close the file and move it from /tmp to a permanent directory."""
        self._accessor.close()
        if self.mode in WRITE_MODES:
            shutil.move(self.temp_path, self.path)
            self.logger.debug(
                f"Sagemaker-Debugger: Wrote {os.path.getsize(self.path)} bytes to file {self.path}"
            )

    def ingest_all(self):
        self._data = self._accessor.read()
        self._datalen = len(self._data)
        self._position = 0

    def read(self, n):
        assert self._position + n <= self._datalen
        res = self._data[self._position : self._position + n]
        self._position += n
        return res

    def has_data(self):
        return self._position < self._datalen
