# First Party
from smdebug.core.logger import get_logger


class TSAccessBase:
    def __init__(self):
        self.logger = get_logger()

    def open(self):
        raise NotImplementedError

    def write(self, _bytes):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def has_data(self):
        raise NotImplementedError
