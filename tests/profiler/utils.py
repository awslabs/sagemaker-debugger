# Standard Library
from datetime import datetime
from pathlib import Path


def get_last_accessed_time(filepath):
    """
    Get the last time that the file at the given filepath was accessed, in the form of a datetime object.
    """
    last_accessed_time = Path(filepath).stat().st_atime
    return datetime.fromtimestamp(last_accessed_time)  # get the last time the config was accessed
