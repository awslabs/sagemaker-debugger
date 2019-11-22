# Standard Library
import os
from enum import Enum

# First Party
from smdebug.core.config_constants import CALLABLE_CACHE_ENV_VAR, DEFAULT_CALLABLE_CACHE
from smdebug.core.modes import ALLOWED_MODES


class CacheType(Enum):
    OFF = 0
    CACHE_PER_MODE = 1
    CLEAR_FOR_EACH_MODE = 2


class CallableCache:
    def __init__(self):
        self._callable_fn_cache = {}  # Maps fetches to callable_fn

        cache_type = os.getenv(CALLABLE_CACHE_ENV_VAR, DEFAULT_CALLABLE_CACHE)

        if cache_type == CacheType.CACHE_PER_MODE.name:
            self.cache_type = CacheType.CACHE_PER_MODE
        elif cache_type == CacheType.CLEAR_FOR_EACH_MODE.name:
            self.cache_type = CacheType.CLEAR_FOR_EACH_MODE
        else:
            self.cache_type = CacheType.OFF

        if self.cache_type == CacheType.CACHE_PER_MODE:
            # create callable cache per mode
            for mode in ALLOWED_MODES:
                self._callable_fn_cache[mode] = {}
        else:
            # cleared cache at the end of each mode
            self._callable_fn_cache = {}

    def change_mode(self):
        if self.cache_type == CacheType.CLEAR_FOR_EACH_MODE:
            self._callable_fn_cache = {}

    def _get_cache(self, mode):
        if self.cache_type == CacheType.CACHE_PER_MODE:
            cache = self._callable_fn_cache[mode]
        elif self.cache_type == CacheType.CLEAR_FOR_EACH_MODE:
            cache = self._callable_fn_cache
        else:
            cache = None
        return cache

    def get_fn(self, mode, fetches):
        cache = self._get_cache(mode)
        if cache is None:
            return None

        fetches.sort(key=lambda x: x.name)
        fetch_hash = tuple(fetches)
        if fetch_hash in cache:
            return cache[fetch_hash]
        else:
            return None

    def cache_fn(self, mode, fetches, callable_fn):
        cache = self._get_cache(mode)
        if cache is None:
            return

        # Cache the fetches mapping to callable
        fetches.sort(key=lambda x: x.name)
        fetch_hash = tuple(fetches)
        if fetch_hash not in cache:
            cache[fetch_hash] = callable_fn
