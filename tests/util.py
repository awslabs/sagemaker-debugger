# Standard Library
import os


class EnvManager(object):
    """Environment variable setter and unsetter via with idiom"""

    def __init__(self, key, val):
        self._key = key
        self._next_val = val
        self._prev_val = None

    def __enter__(self):
        self._prev_val = os.environ.get(self._key)
        os.environ[self._key] = self._next_val

    def __exit__(self, ptype, value, trace):
        if self._prev_val:
            os.environ[self._key] = self._prev_val
        else:
            del os.environ[self._key]
