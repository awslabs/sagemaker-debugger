# Standard Library

import bisect
import json
import os

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler, is_s3
from smdebug.core.logger import get_logger
from smdebug.core.utils import list_files_in_directory
from smdebug.profiler.profiler_constants import (
    DEFAULT_PREFIX,
    HOROVODTIMELINE_PREFIX,
    MODELTIMELINE_SUFFIX,
    PYTHONTIMELINE_SUFFIX,
    TENSORBOARDTIMELINE_SUFFIX,
    TIME_BUFFER,
)
from smdebug.profiler.tf_profiler_parser import (
    HorovodProfilerEvents,
    SMProfilerEvents,
    TensorboardProfilerEvents,
)
from smdebug.profiler.utils import TimeUnits, convert_utc_timestamp_to_microseconds


class MetricsReader:
    def __init__(self):
        self.prefix = DEFAULT_PREFIX
        self.logger = get_logger("smdebug-profiler")
        self._SMEventsParser = SMProfilerEvents()
        self._TBEventsParser = TensorboardProfilerEvents()
        self._HorovordEventsParser = HorovodProfilerEvents()
        # This is a set of parsed event files. The entry is made into this file only if the complete file is read.
        self._parsed_files = set()
        self._timestamp_to_filename = dict()

    def _get_node_id_from_filename(self, filename):
        filename = filename.split("/")[-1]
        return int(filename.split("_")[1])

    def _get_end_timestamp_from_filename(self, filename):
        filename = filename.split("/")[-1]
        return int(filename.split("_")[0])

    """
    The function returns the timestamp of last available file.
    This timestamp indicates users can query the events up to this timestamp to gauge
    """

    def get_timestamp_of_latest_available_file(self):
        return (
            sorted(self._timestamp_to_filename.keys())[-1]
            if len(self._timestamp_to_filename) > 0
            else 0
        )

    """
    Return the tracefiles that were written during the given range. If use_buffer is True, we will consider adding a
    buffer of TIME_BUFFER microseconds to increase the time range. This is done because the events are written to the file
    after they end. It is possible that an event would have stared within the window of start and end, however it
    did not complete at or before 'end' time. Hence the event will not appear in the tracefile that corresponds to
    'end' timestamp. It will appear in the future event file.
    We will also add a buffer for the 'start' i.e. we will look for tracefiles that were written prior to 'start'.
    Those files might contain 'B' type events that had started prior to 'start'
    """

    def _get_trace_files_in_the_range(
        self, start_time_microseconds, end_time_microseconds, use_buffer=True
    ):
        # increase the time range using TIME_BUFFER
        if use_buffer:
            start_time_microseconds = start_time_microseconds - TIME_BUFFER
            end_time_microseconds = end_time_microseconds + TIME_BUFFER

        # If the end_time_microseconds is less than the timestamp of the latest available tracefile, we do not need
        # to poll for new event files and refresh the list from directory or S3 bucket.
        if self.get_timestamp_of_latest_available_file() <= end_time_microseconds:
            self.refresh_event_file_list()

        timestamps = sorted(self._timestamp_to_filename.keys())

        # Find the timestamp that is greater than or equal start_time_microseconds. The tracefile corresponding to
        # that timestamp will contain events that are active during start_time_microseconds
        lower_bound_timestamp = bisect.bisect_left(timestamps, start_time_microseconds)

        # Find the timestamp that is immediate right to the end_time_microseconds. The tracefile corresponding to
        # that timestamp will contain events that are active during end_time_microseconds.
        upper_bound_timestamp = bisect.bisect_right(timestamps, end_time_microseconds)

        event_files = list()
        for index in timestamps[lower_bound_timestamp:upper_bound_timestamp]:
            event_files.append(self._timestamp_to_filename[index])
        return event_files

    """
    The function returns the right event parser for given file name
    1. For Filename containing 'pythontimeline.json'  -> SMEventsParser
    2. For Filename containing 'model_timeline.json'  -> SMEventsParser
    3. For Filename containing 'tensorboard' (TBD) -> TensorboardProfilerEvents
    4. For Filename containing 'Horovod' (TBD) -> 'HorovodProfilerEvents
    """

    def _get_event_parser(self, filename):
        if PYTHONTIMELINE_SUFFIX in filename:
            return self._SMEventsParser
        if MODELTIMELINE_SUFFIX in filename:
            return self._SMEventsParser
        if TENSORBOARDTIMELINE_SUFFIX in filename:
            return self._TBEventsParser
        if HorovodProfilerEvents in filename:
            return self._HorovordEventsParser

    """
    This function queries the files that are currently available in the directory (for local mode) or in S3 for download.
    It rebuilds the map of timestamp to filename.
    """

    def refresh_event_file_list(self):
        pass

    """
    The function returns the events that have recorded within the given time range.
    This is currently non-blocking call. The function will download (or parse) the tracefiles that are available
    for the given time range. It is possible that events are recorded during training but are not available for
    download.
    TODO: Implement caching to avoid repeat download
    TODO: Implement blocking call to wait for files to be available for download.
    """

    def get_events(self, start_time, end_time, unit=TimeUnits.MICROSECONDS):
        start_time = convert_utc_timestamp_to_microseconds(start_time, unit)
        end_time = convert_utc_timestamp_to_microseconds(end_time, unit)

        event_files = self._get_trace_files_in_the_range(start_time, end_time)

        # Download files and parse the events
        self.parse_event_files(event_files)

        """
        We might have recorded events from different sources within this timerange.
        we will get the events from the relevant event parsers and merge them before returning.
        """
        result = []
        for eventParser in [self._SMEventsParser, self._TBEventsParser, self._HorovordEventsParser]:
            range_events = eventParser.get_events_within_time_range(
                start_time, end_time, unit=TimeUnits.MICROSECONDS
            )
            result.extend(range_events)

        return result

    def parse_event_files(self, event_files):
        pass


class LocalMetricsReader(MetricsReader):
    def __init__(self, trace_root_folder):
        self.trace_root_folder = trace_root_folder
        super().__init__()

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        path = os.path.expanduser(self.trace_root_folder)
        event_dir = os.path.join(path, DEFAULT_PREFIX, "")
        event_regex = r"(.+)\.(json|csv)$"
        event_files = list_files_in_directory(event_dir, file_regex=event_regex)
        for event_file in event_files:
            timestamp = self._get_end_timestamp_from_filename(event_file)
            self._timestamp_to_filename[timestamp] = event_file

    """
    The function opens and reads the event files if they are not already parsed.
    For local metrics reader, we are currently assuming that the downloaded event file is a complete file.
    """

    def parse_event_files(self, event_files):
        for event_file in event_files:
            if event_file not in self._parsed_files:
                self._get_event_parser(event_file).read_events_from_file(event_file)
                self._parsed_files.add(event_file)


class S3MetricsReader(MetricsReader):
    """
    The s3_trial_path points to a s3 folder in which the tracefiles are stored. e.g.
    s3://my_bucket/experiment_base_folder
    """

    def __init__(self, s3_trial_path):
        super().__init__()
        s3, bucket_name, base_folder = is_s3(s3_trial_path)
        if not s3:
            self.logger.error(
                "The trial path is expected to be S3 path e.g. s3://bucket_name/trial_folder"
            )
        else:
            self.bucket_name = bucket_name
            self.base_folder = base_folder
            self.prefix = os.path.join(self.base_folder, self.prefix, "")

    """
    The function opens and reads the event files if they are not already parsed.
    For S3 metrics reader, we are currently downloading the entire event file. Currently, we will add this file to a
    _parsed_file set assuming that the file is complete and it won't be updated on S3 later. However it is
    possible that the event file has not reached it's maximum size and will be updated later.
    TODO: Check the downloaded size and add the file to _parsed_files only if the file with maximum size is
    downloaded and parsed.
    """

    def parse_event_files(self, event_files):
        for event_file in event_files:
            if event_file not in self._parsed_files:
                file_read_request = ReadObjectRequest(path=event_file)
                event_data = S3Handler.get_object(file_read_request)
                event_string = event_data.decode("utf-8")
                json_data = json.loads(event_string)
                self._get_event_parser(event_file).read_events_from_json_data(json_data)
                self._parsed_files.add(event_file)

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        list_dir = ListRequest(Bucket=self.bucket_name, Prefix=self.prefix, StartAfter=self.prefix)

        event_files = [x for x in S3Handler.list_prefix(list_dir) if "json" in x]
        for event_file in event_files:
            timestamp = self._get_end_timestamp_from_filename(event_file)
            self._timestamp_to_filename[timestamp] = f"s3://{self.bucket_name}/{event_file}"
