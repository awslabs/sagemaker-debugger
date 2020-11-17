"""
The TimeUnit enum is to be used while querying the events within timerange or at a given timestamp
The Enum will indicate the unit in which timestamp is provided.
"""
# Standard Library
import os
import re
import shutil
import time
from datetime import datetime
from distutils.util import strtobool
from enum import Enum
from pathlib import Path

# Third Party
from botocore.exceptions import ClientError

# First Party
from smdebug.core.access_layer.file import SMDEBUG_TEMP_PATH_SUFFIX, TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.access_layer.s3handler import ListRequest, S3Handler, is_s3
from smdebug.core.logger import get_logger
from smdebug.core.utils import ensure_dir, get_node_id
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    HOROVODTIMELINE_SUFFIX,
    SMDATAPARALLELTIMELINE_SUFFIX,
)

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


def convert_utc_datetime_to_microseconds(timestamp_datetime: datetime):
    return convert_utc_timestamp_to_microseconds(
        timestamp_datetime.timestamp(), unit=TimeUnits.SECONDS
    )


def is_valid_tfprof_tracefilename(filename: str) -> bool:
    """
    Ensure that the tracefilename has a valid format.
    $ENV_BASE_FOLDER/framework/tensorflow/detailed_profiling/$START_TIME_YYYYMMDDHR/$STEP_NUM/plugins/profile/$HOSTNAME.trace.json.gz

    The filename should have extension trace.json.gz

    """
    return filename.endswith("trace.json.gz") and "tensorflow/detailed_profiling" in filename


def is_valid_tracefilename(filename: str) -> bool:
    """
    Ensure that the tracefilename has a valid format.
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTENDTIMEUTCINEPOCH_{$ENV_NODE_ID}_model_timeline.json

    The filename should have extension .json
    The filename should have minimum 3 fields viz. $FILEEVENTENDTIMEUTCINEPOCH, {$ENV_NODE_ID} and filetype.

    """
    if filename.endswith(".json") and "pevents" in filename:
        if len(filename.split("_")) >= 3:
            return True
    logger.debug(f"The file {filename} is not a valid tracefile.")
    return False


def get_node_id_from_tracefilename(filename: str) -> str:
    """
    The tracefile has a file name format:
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTENDTIMEUTCINEPOCH_{$ENV_NODE_ID}_model_timeline.json

    The function extracts and returns the {$ENV_NODE_ID} from file.
    """
    if is_valid_tracefilename(filename):
        filename = filename.split("/")[-1]
        return filename.split("_")[1]
    else:
        node_id, _, _ = read_tf_profiler_metadata_file(filename)
        return node_id


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
    if is_valid_tracefilename(filename):
        filename = filename.split("/")[-1]
        return int(filename.split("_")[0])
    else:
        _, _, timestamp = read_tf_profiler_metadata_file(filename)
        return int(timestamp)


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
    dt = datetime.utcfromtimestamp(us_since_epoch / 1e6)
    return dt.strftime("%Y-%m-%dT%H:%M:%S:%f")


def ns_since_epoch_to_human_readable_time(ns_since_epoch):
    dt = datetime.utcfromtimestamp(ns_since_epoch / 1e9)
    return dt.strftime("%Y-%m-%dT%H:%M:%S:%f")


def write_tf_profiler_metadata_file(file_path):
    if not file_path.endswith(".metadata"):
        return
    s3, bucket_name, key_name = is_s3(file_path)
    if s3:
        writer = TSAccessS3(bucket_name, key_name, binary=False)
    else:
        writer = TSAccessFile(file_path, "a+")
    writer.flush()
    try:
        writer.close()
    except OSError:
        """
        In the case of distributed training in local mode,
        another worker may have already moved the END_OF_JOB file
        from the /tmp directory.
        """


def read_tf_profiler_metadata_file(file_path):
    if not is_valid_tfprof_tracefilename(file_path):
        return "", "0", "0"
    s3, bucket_name, key_name = is_s3(file_path)
    if s3:
        try:
            folder_name = "/".join(key_name.split("/")[:-4])
            request = ListRequest(bucket_name, folder_name)
            file_available = S3Handler.list_prefixes([request])
            if len(file_available) > 0:
                metadata_filename = list(filter(lambda x: ".metadata" in x, file_available[0]))
                if len(metadata_filename) > 0:
                    metadata_filename = metadata_filename[0]
                    metadata_filename = metadata_filename.split("/")[-1]
                    node_id, start, end = str(metadata_filename).split("_")
                    return node_id, start, end.split(".")[0]
                else:
                    return "", "0", "0"
            else:
                return "", "0", "0"
        except ClientError as ex:
            status_code = ex.response["ResponseMetadata"]["HTTPStatusCode"]
            logger.info(f"Client error occurred : {ex}")
            if status_code.startswith("4"):
                raise ex
            else:
                return "", "0", "0"
    else:
        folder_name = "/".join(file_path.split("/")[:-4])
        metadata_filename = list(Path(folder_name).rglob("*.metadata"))
        if len(metadata_filename) > 0:
            metadata_filename = metadata_filename[0].name
            node_id, start, end = str(metadata_filename).split("_")
            return node_id, start, end.split(".")[0]
        else:
            return "", "0", "0"


def stop_tf_profiler(tf_profiler, log_dir, start_time_us):
    from smdebug.core.locations import TraceFileLocation

    tf_profiler.stop()
    metadata_file = TraceFileLocation.get_tf_profiling_metadata_file(
        log_dir, start_time_us, time.time() * CONVERT_TO_MICROSECS
    )
    write_tf_profiler_metadata_file(metadata_file)


def start_smdataparallel_profiler(smdataparallel, base_dir):
    if smdataparallel:
        from smdistributed.dataparallel import start_profiler

        smdataparallel_temp_file = os.path.join(
            base_dir, f"{get_node_id()}_{SMDATAPARALLELTIMELINE_SUFFIX}{SMDEBUG_TEMP_PATH_SUFFIX}"
        )
        ensure_dir(smdataparallel_temp_file)
        start_profiler(smdataparallel_temp_file, append_rank=False)


def stop_smdataparallel_profiler(smdataparallel, base_dir):
    from smdebug.core.locations import TraceFileLocation

    if smdataparallel:
        from smdistributed.dataparallel import stop_profiler

        smdataparallel_temp_file = os.path.join(
            base_dir, f"{get_node_id()}_{SMDATAPARALLELTIMELINE_SUFFIX}{SMDEBUG_TEMP_PATH_SUFFIX}"
        )
        stop_profiler()
        new_file_name = TraceFileLocation.get_file_location(
            time.time() * CONVERT_TO_MICROSECS, base_dir, suffix=SMDATAPARALLELTIMELINE_SUFFIX
        )
        ensure_dir(new_file_name)
        if os.path.exists(smdataparallel_temp_file):
            shutil.move(smdataparallel_temp_file, new_file_name)
