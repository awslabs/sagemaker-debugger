# Standard Library

import bisect
import json
import os

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler, is_s3
from smdebug.profiler.metrics_reader_base import MetricsReaderBase
from smdebug.profiler.profiler_constants import (
    DEFAULT_SYSTEM_PROFILER_PREFIX,
    ENV_TIME_BUFFER,
    TIME_BUFFER_DEFAULT,
)
from smdebug.profiler.system_profiler_file_parser import ProfilerSystemEvents
from smdebug.profiler.utils import get_utctimestamp_us_since_epoch_from_system_profiler_file


class SystemMetricsReader(MetricsReaderBase):
    def __init__(self, use_in_memory_cache=False):
        super().__init__(use_in_memory_cache)
        self.prefix = DEFAULT_SYSTEM_PROFILER_PREFIX
        self._SystemProfilerEventParser = ProfilerSystemEvents()
        self._event_parsers = [self._SystemProfilerEventParser]

    """
    Return the profiler system event files that were written during the given range. If use_buffer is True, we will consider adding a
    buffer of TIME_BUFFER_DEFAULT microseconds to increase the time range. This is done because the events are written to the
    file after they end. It is possible that an event would have started within the window of start and end, however it
    did not complete at or before 'end' time. Hence the event will not appear in the event file that corresponds to
    'end' timestamp. It will appear in the future event file.
    We will also add a buffer for the 'start' i.e. we will look for event files that were written prior to 'start'.
    Those files might contain 'B' type events that had started prior to 'start'
    """

    def _get_event_files_in_the_range(
        self, start_time_microseconds, end_time_microseconds, use_buffer=True
    ):
        # increase the time range using TIME_BUFFER_DEFAULT
        if use_buffer:
            time_buffer = os.getenv(ENV_TIME_BUFFER, TIME_BUFFER_DEFAULT)
            start_time_microseconds = start_time_microseconds - time_buffer
            end_time_microseconds = end_time_microseconds + time_buffer

        """
        We need to intelligently detect whether we need to refresh the list of available event files.
        Approach 1: Keep the start prefix for S3, 'x' minutes (say 5) lagging behind the last available timestamp.
        This will cover for the case where a node or writer is not efficient enough to upload the files to S3
        immediately. For local mode we may have to walk the directory every time.
        This is currently implemented by computing the start prefix and TRAILING DURATION.
        TODO:
        Approach 2: If we can know the expected number of files per node and per writer, we can intelligently wait
        for that type of file for certain amount of time.
        """

        """
        In case of S3, we will refresh the event file list if the requested end timestamp is less than the timestamp
        of _startAfterPrefix.
        In case of local mode, the event file list will be refreshed if the end timestamp is not less than the last
        available timestamp
        """

        if self._startAfter_prefix is not "":
            if end_time_microseconds >= get_utctimestamp_us_since_epoch_from_system_profiler_file(
                self._startAfter_prefix
            ):
                self.refresh_event_file_list()
        else:
            if end_time_microseconds >= self.get_timestamp_of_latest_available_file():
                self.refresh_event_file_list()

        timestamps = sorted(self._timestamp_to_filename.keys())

        # Find the timestamp that is smaller than or equal start_time_microseconds. The event file corresponding to
        # that timestamp will contain events that are active during start_time_microseconds
        lower_bound_timestamp_index = bisect.bisect_right(timestamps, start_time_microseconds)
        if lower_bound_timestamp_index > 0:
            lower_bound_timestamp_index -= 1

        # Find the timestamp that is immediate right to the end_time_microseconds. The event file corresponding to
        # that timestamp will contain events that are active during end_time_microseconds.
        upper_bound_timestamp_index = bisect.bisect_left(timestamps, end_time_microseconds)

        event_files = list()
        for index in timestamps[lower_bound_timestamp_index : upper_bound_timestamp_index + 1]:
            event_files.extend(self._timestamp_to_filename[index])
        return event_files

    def _get_event_parser(self, filename):
        return self._SystemProfilerEventParser

    def _get_timestamp_from_filename(self, event_file):
        return get_utctimestamp_us_since_epoch_from_system_profiler_file(event_file)

    def _get_event_file_regex(self):
        return r"(.+)\.json"


class LocalSystemMetricsReader(SystemMetricsReader):
    """
    The metrics reader is created with root folder in which the system event files are stored.
    """

    def __init__(self, trace_root_folder, use_in_memory_cache=False):
        self.trace_root_folder = trace_root_folder
        super().__init__(use_in_memory_cache)
        # Pre-build the file list so that user can query get_timestamp_of_latest_available_file()
        # and get_current_time_range_for_event_query
        self.refresh_event_file_list()

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        self._refresh_event_file_list_local_mode(self.trace_root_folder)

    def parse_event_files(self, event_files):
        self._parse_event_files_local_mode(event_files)


class S3SystemMetricsReader(SystemMetricsReader):
    """
    The s3_trial_path points to a s3 folder in which the system metric event files are stored. e.g.
    s3://my_bucket/experiment_base_folder
    """

    def __init__(self, s3_trial_path, use_in_memory_cache=False):
        super().__init__(use_in_memory_cache)
        s3, bucket_name, base_folder = is_s3(s3_trial_path)
        if not s3:
            self.logger.error(
                "The trial path is expected to be S3 path e.g. s3://bucket_name/trial_folder"
            )
        else:
            self.bucket_name = bucket_name
            self.base_folder = base_folder
            self.prefix = os.path.join(self.base_folder, self.prefix, "")
        # Pre-build the file list so that user can query get_timestamp_of_latest_available_file()
        # and get_current_time_range_for_event_query
        self.refresh_event_file_list()

    def parse_event_files(self, event_files):
        file_read_requests = []
        event_files_to_read = []

        for event_file in event_files:
            if event_file not in self._parsed_files:
                event_files_to_read.append(event_file)
                file_read_requests.append(ReadObjectRequest(path=event_file))

        event_data_list = S3Handler.get_objects(file_read_requests)
        for event_data, event_file in zip(event_data_list, event_files_to_read):
            event_string = event_data.decode("utf-8")
            event_items = event_string.split("\n")
            event_items.remove("")
            for item in event_items:
                event = json.loads(item)
                self._SystemProfilerEventParser.read_event_from_dict(event)
            self._parsed_files.add(event_file)

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        list_dir = ListRequest(
            Bucket=self.bucket_name,
            Prefix=self.prefix,
            StartAfter=self._startAfter_prefix if self._startAfter_prefix else self.prefix,
        )
        self._refresh_event_file_list_s3_mode(list_dir)
