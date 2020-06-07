# Standard Library

import bisect
import json
import os

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler, is_s3
from smdebug.core.logger import get_logger
from smdebug.core.utils import (
    get_node_id_from_tracefilename,
    get_timestamp_from_tracefilename,
    list_files_in_directory,
)
from smdebug.profiler.profiler_constants import (
    DEFAULT_PREFIX,
    ENV_TIME_BUFFER,
    ENV_TRAIILING_DURATION,
    HOROVODTIMELINE_PREFIX,
    MODELTIMELINE_SUFFIX,
    PYTHONTIMELINE_SUFFIX,
    TENSORBOARDTIMELINE_SUFFIX,
    TIME_BUFFER_DEFAULT,
    TRAILING_DURATION_DEFAULT,
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

        # The startAfter_prefix is used in ListPrefix call to poll for available tracefiles in the S3 bucket. The
        # prefix lags behind the last polled tracefile by tunable trailing duration. This is to ensure that we do not
        # miss a
        # tracefile corresponding to timestamp earlier than last polled timestamp but arrived after we had polled.

        self._startAfter_prefix = ""

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
    The following function returns the time range for which the tracefiles are currently available in S3 or local
    directory. Users can still query for events for the window greater than this range. In that case, the reader will
    query S3 or local directory to check the tracefiles are available.
    """

    def get_current_time_range_for_event_query(self):
        timestamps = self._timestamp_to_filename.keys()
        return (timestamps[0], timestamps[-1]) if len(timestamps) > 0 else (0, 0)

    """
    Return the tracefiles that were written during the given range. If use_buffer is True, we will consider adding a
    buffer of TIME_BUFFER_DEFAULT microseconds to increase the time range. This is done because the events are written to the
    file after they end. It is possible that an event would have stared within the window of start and end, however it
    did not complete at or before 'end' time. Hence the event will not appear in the tracefile that corresponds to
    'end' timestamp. It will appear in the future event file.
    We will also add a buffer for the 'start' i.e. we will look for tracefiles that were written prior to 'start'.
    Those files might contain 'B' type events that had started prior to 'start'
    """

    def _get_trace_files_in_the_range(
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

        if end_time_microseconds >= self.get_timestamp_of_latest_available_file() or end_time_microseconds >= get_timestamp_from_tracefilename(
            self._startAfter_prefix
        ):
            self.refresh_event_file_list()

        timestamps = sorted(self._timestamp_to_filename.keys())

        # Find the timestamp that is greater than or equal start_time_microseconds. The tracefile corresponding to
        # that timestamp will contain events that are active during start_time_microseconds
        lower_bound_timestamp = bisect.bisect_left(timestamps, start_time_microseconds)

        # Find the timestamp that is immediate right to the end_time_microseconds. The tracefile corresponding to
        # that timestamp will contain events that are active during end_time_microseconds.
        upper_bound_timestamp = bisect.bisect_left(timestamps, end_time_microseconds)

        event_files = list()
        for index in timestamps[lower_bound_timestamp : upper_bound_timestamp + 1]:
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
    The function will download (or parse) the tracefiles that are available
    for the given time range. It is possible that events are recorded during training but are not available for
    download.
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
        # Pre-build the file list so that user can query get_timestamp_of_latest_available_file() and get_current_time_range_for_event_query
        self.refresh_event_file_list()

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        path = os.path.expanduser(self.trace_root_folder)
        event_dir = os.path.join(path, DEFAULT_PREFIX, "")
        event_regex = r"(.+)\.(json|csv)$"
        event_files = list_files_in_directory(event_dir, file_regex=event_regex)
        for event_file in event_files:
            timestamp = get_timestamp_from_tracefilename(event_file)
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
        # Pre-build the file list so that user can query get_timestamp_of_latest_available_file() and get_current_time_range_for_event_query
        self.refresh_event_file_list()

    """
    The function opens and reads the event files if they are not already parsed.
    For S3 metrics reader, we are currently downloading the entire event file. Currently, we will add this file to a
    _parsed_file set assuming that the file is complete and it won't be updated on S3 later. However it is
    possible that the event file has not reached it's maximum size and will be updated later.
    TODO: Check the downloaded size and add the file to _parsed_files only if the file with maximum size is
    downloaded and parsed.
    """

    def parse_event_files(self, event_files):
        file_read_requests = []
        for event_file in event_files:
            if event_file not in self._parsed_files:
                file_read_requests.append(ReadObjectRequest(path=event_file))

        event_data_list = S3Handler.get_objects(file_read_requests)
        for event_data, event_file in zip(event_data_list, event_files):
            event_string = event_data.decode("utf-8")
            json_data = json.loads(event_string)
            node_id = get_node_id_from_tracefilename(event_file)
            self._get_event_parser(event_file).read_events_from_json_data(json_data, node_id)
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
        event_files = [x for x in S3Handler.list_prefix(list_dir) if "json" in x]
        for event_file in event_files:
            timestamp = get_timestamp_from_tracefilename(event_file)
            self._timestamp_to_filename[timestamp] = f"s3://{self.bucket_name}/{event_file}"
        self.update_start_after_prefix()

    """
    It is possible that tracefiles from different nodes to arrive in S3 in different order. For example, Even if t1
    > t2, a tracefile with timestamp "t1" can arrive in S3 before the tracefile with timestamp "t2". If we list the
    prefix only on the basis of last arrived file (i.e. t1) we will miss the file for t2. Therefore, we will set the
    start prefix to a timestamp that is trailing behind the last timestamp by 'trailing duration'. This will ensure
    that we will attempt to get tracefiles with older timestamp even if they arrive late.
    """

    def update_start_after_prefix(self):
        trailiing_duration = os.getenv(ENV_TRAIILING_DURATION, TRAILING_DURATION_DEFAULT)
        sorted_timestamps = sorted(self._timestamp_to_filename.keys())
        last_timestamp_available = sorted_timestamps[-1]
        trailing_timestamp = last_timestamp_available - trailiing_duration
        # Get the timestamp that is closely matching the trailing_timestamp
        trailing_timestamp = sorted_timestamps[
            bisect.bisect_left(sorted_timestamps, trailing_timestamp)
        ]
        self._startAfter_prefix = self._timestamp_to_filename[trailing_timestamp]
        s3, bucket_name, self._startAfter_prefix = is_s3(self._startAfter_prefix)
