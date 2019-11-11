# First Party
from tornasole.core.logger import get_logger


class TSAccessBase:
    def __init__(self):
        self.logger = get_logger()
        pass

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
