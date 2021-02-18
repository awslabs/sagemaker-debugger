# Standard Library
import types

# First Party
from smdebug.core.logger import DuplicateLogFilter
from smdebug.core.utils import get_logger


def test_dup_filter():
    logger = get_logger()
    dup_filter = None

    for _filter in logger.filters:
        if isinstance(_filter, DuplicateLogFilter):
            dup_filter = _filter
    dup_filter.test_counter = 0

    def filter(self, record):
        self.msgs[record.msg] += 1
        if self.msgs[record.msg] > self.repeat_threshold:
            return True
        else:
            self.test_counter += 1
            return False

    dup_filter.filter = types.MethodType(filter, dup_filter)

    for _ in range(10):
        logger.warning("I love spam musubi")

    assert dup_filter.test_counter == 5
