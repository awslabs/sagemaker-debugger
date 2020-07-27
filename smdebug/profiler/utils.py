"""
The TimeUnit enum is to be used while querying the events within timerange or at a given timestamp
The Enum will indicate the unit in which timestamp is provided.
"""
# Standard Library
import re
from datetime import datetime
from distutils.util import strtobool
from enum import Enum

# First Party
from smdebug.core.logger import get_logger

logger = get_logger()


class TimeUnits(Enum):
    SECONDS = 1
    MILLISECONDS = 2
    MICROSECONDS = 3
    NANOSECONDS = 4


"""
The function assumes that the correct enum is provided.
"""


def convert_utc_timestamp_to_nanoseconds(timestamp, unit=TimeUnits.MICROSECONDS):
    if unit == TimeUnits.SECONDS:
        return int(timestamp * 1000 * 1000 * 1000)
    if unit == TimeUnits.MILLISECONDS:
        return int(timestamp * 1000 * 1000)
    if unit == TimeUnits.MICROSECONDS:
        return int(timestamp * 1000)
    if unit == TimeUnits.NANOSECONDS:
        return timestamp


def convert_utc_timestamp_to_microseconds(timestamp, unit=TimeUnits.MICROSECONDS):
    if unit == TimeUnits.SECONDS:
        return int(timestamp * 1000 * 1000)
    if unit == TimeUnits.MILLISECONDS:
        return int(timestamp * 1000)
    if unit == TimeUnits.MICROSECONDS:
        return int(timestamp)
    if unit == TimeUnits.NANOSECONDS:
        return timestamp / 1000


def convert_utc_timestamp_to_seconds(timestamp, unit=TimeUnits.MICROSECONDS):
    if unit == TimeUnits.SECONDS:
        return int(timestamp)
    if unit == TimeUnits.MILLISECONDS:
        return int(timestamp / 1000)
    if unit == TimeUnits.MICROSECONDS:
        return int(timestamp / 1000 / 1000)
    if unit == TimeUnits.NANOSECONDS:
        return timestamp / 1000 / 1000 / 1000


"""
The function assumes that the object of datetime is provided.
"""


def convert_utc_datetime_to_nanoseconds(timestamp_datetime: datetime):
    return convert_utc_timestamp_to_nanoseconds(
        timestamp_datetime.timestamp(), unit=TimeUnits.SECONDS
    )


def is_valid_tracefilename(filename: str) -> bool:
    """
    Ensure that the tracefilename has a valid format.
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTENDTIMEUTCINEPOCH_{$ENV_NODE_ID}_model_timeline.json

    The filename should have extension .json
    The filename should have minimum 3 fields viz. $FILEEVENTENDTIMEUTCINEPOCH, {$ENV_NODE_ID} and filetype.

    """
    if filename.endswith(".json"):
        if len(filename.split("_")) >= 3:
            return True
    logger.error(f"The file {filename} is not a valid tracefile.")
    return False


def get_node_id_from_tracefilename(filename: str) -> str:
    """
    The tracefile has a file name format:
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTENDTIMEUTCINEPOCH_{$ENV_NODE_ID}_model_timeline.json

    The function extracts and returns the {$ENV_NODE_ID} from file.
    """
    filename = filename.split("/")[-1]
    return filename.split("_")[1] if is_valid_tracefilename(filename) else ""


def get_node_id_from_system_profiler_filename(filename: str) -> str:
    """
    The system metric has a file name format:
    /profiler-output/system/incremental/{$TIMESTAMP}.${NODE_ID}.json
    Example: /profiler-output/system/incremental/2020060500/1591160699.algo-1.json

    The function extracts and returns the {$NODE_ID} from file.
    """
    if validate_system_profiler_file(filename):
        filename = filename.split("/")[-1]
        return filename.split(".")[1]
    return None


def get_timestamp_from_tracefilename(filename) -> int:
    """
    The tracefile has a file name format:
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTENDTIMEUTCINEPOCH_{$ENV_NODE_ID}_model_timeline.json

    The function extracts and returns the $FILEEVENTENDTIMEUTCINEPOCH from file. The $FILEEVENTENDTIMEUTCINEPOCH
    represents the timestamp of last event written to the tracefile.
    The timestamps are used to determine whether an event is available in this this file. If the file name is not
    valid, we will written timestamp as 0.
    """
    filename = filename.split("/")[-1]
    return int(filename.split("_")[0] if is_valid_tracefilename(filename) else "0")


def get_utctimestamp_us_since_epoch_from_system_profiler_file(filename) -> int:
    """
    The system metric file has a file name format:
    <training job name>/profiler-output/system/incremental/<timestamp of full minute>.<algo-n>.json
    Example: /profiler-output/system/incremental/2020060500/1591160699.algo-1.json

    The function extracts and returns the <timestamp of full minute> in microseconds from filename.
    """
    if validate_system_profiler_file(filename):
        filename = filename.split("/")[-1]
        return int(filename.split(".")[0]) * 1000 * 1000
    return None


def validate_system_profiler_file(filename) -> bool:
    filename_regex = re.compile(".+/system/.+/(\d{10}).algo-\d+.json")
    stamp = re.match(filename_regex, filename)
    if stamp is None:
        logger.debug(f"Invalid System Profiler File Found: {filename}, not able to get timestamp.")
        return False
    return True


def str2bool(v):
    if isinstance(v, bool):
        return v
    else:
        return bool(strtobool(v))


def us_since_epoch_to_human_readable_time(us_since_epoch):
    dt = datetime.fromtimestamp(us_since_epoch / 1e6)
    return dt.strftime("%Y-%m-%dT%H:%M:%S:%f")


class CaseInsensitiveConfig:
    def __init__(self, config):
        """
        Utils class that builds on top of the native Python dictionary to make keys case insensitive.
        :param config The underlying dictionary whose keys should be case insensitive.
        """
        self._config = self._convert_keys_to_uppercase(config)

    def _convert_keys_to_uppercase(self, config):
        """
        Recursively converts all keys in the provided dictionary to be upper case. This generates a new dictionary, so
        the provided dictionary is not modified.
        """
        return {
            key.upper(): self._convert_keys_to_uppercase(value)
            if isinstance(value, dict)
            else value
            for key, value in config.items()
        }

    def get(self, key, default_value=None):
        """
        Higher level version of the native dictionary get function to convert the provided key to upper case first.
        """
        value = self._config.get(key.upper(), default_value)
        return CaseInsensitiveConfig(value) if isinstance(value, dict) else value
