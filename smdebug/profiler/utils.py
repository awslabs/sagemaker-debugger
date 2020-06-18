"""
The TimeUnit enum is to be used while querying the events within timerange or at a given timestamp
The Enum will indicate the unit in which timestamp is provided.
"""
# Standard Library
from datetime import datetime
from enum import Enum


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
