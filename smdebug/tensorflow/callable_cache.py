# Standard Library
import os
from enum import Enum

# First Party
from smdebug.core import logger
from smdebug.core.config_constants import CALLABLE_CACHE_ENV_VAR, DEFAULT_CALLABLE_CACHE
from smdebug.core.modes import ALLOWED_MODES


class CacheType(Enum):
    OFF = 0
    CACHE_PER_MODE = 1  # maintain callable_fn cache per mode separately
    CLEAR_FOR_EACH_MODE = 2  # maintain one cache which gets cleared each time mode changes


class CallableCache:
    """
    Helper class to cache the callable_fn in Keras
    Ref: https://github.com/tensorflow/tensorflow/blob/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b/tensorflow/python/keras/backend.py#L3473

    Each time we change fetches, we check if we have a computed callable_fn for that fetches,
    if so we trick Keras into thinking the last step's fetches is the same as
    the new fetches we're sending, and set the callable_fn which was in our cache.

    Note that if the feeds change, keras will still compute the cache.
    We need to change the cache on mode change, because the earlier callable is no longer relevant for the new mode.
    It should start from None again. There might be tensors whose placeholders won't be filled if we try to use
    old callable
    """

    def __init__(self):
        self._callable_fn_cache = {}  # Maps fetches to callable_fn

        cache_type = os.getenv(CALLABLE_CACHE_ENV_VAR, DEFAULT_CALLABLE_CACHE)

        if cache_type == CacheType.CACHE_PER_MODE.name:
            self.cache_type = CacheType.CACHE_PER_MODE
        elif cache_type == CacheType.CLEAR_FOR_EACH_MODE.name:
            self.cache_type = CacheType.CLEAR_FOR_EACH_MODE
        else:
            self.cache_type = CacheType.OFF

        logger.get_logger().debug(f"Created callable_fn cache of type {self.cache_type.name}")

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
        return cache.get(tuple(fetches), None)

    def cache_fn(self, mode, fetches, callable_fn):
        cache = self._get_cache(mode)
        if cache is None:
            return

        # Cache the fetches mapping to callable
        fetches.sort(key=lambda x: x.name)
        fetch_hash = tuple(fetches)
        cache[fetch_hash] = callable_fn
