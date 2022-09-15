# First Party
from smdebug.core.logger import get_logger
from smdebug.exceptions import SMDebugNotImplementedError


class TSAccessBase:
    def __init__(self):
        self.logger = get_logger()

    def open(self):
        raise SMDebugNotImplementedError

    def write(self, _bytes):
        raise SMDebugNotImplementedError

    def flush(self):
        raise SMDebugNotImplementedError

    def close(self):
        raise SMDebugNotImplementedError

    def has_data(self):
        raise SMDebugNotImplementedError
