# Standard Library
import types

# First Party
from smdebug.core.config_constants import LOG_DUPLICATION_THRESHOLD
from smdebug.core.logger import DuplicateLogFilter
from smdebug.core.utils import get_logger


def test_dup_filter():
    logger = get_logger()
    dup_filter = None

    for _filter in logger.filters:
        if isinstance(_filter, DuplicateLogFilter):
            dup_filter = _filter
    dup_filter.test_counter = 0

    dup_filter.old_filter = dup_filter.filter

    def filter(self, record):
        if self.old_filter(record):
            self.test_counter += 1
            return True
        else:
            return False

    dup_filter.filter = types.MethodType(filter, dup_filter)

    for _ in range(10):
        logger.warning("I love spam musubi")

    assert dup_filter.test_counter == LOG_DUPLICATION_THRESHOLD
