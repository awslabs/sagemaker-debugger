# Standard Library

import bisect
import json
import os

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler, is_s3
from smdebug.profiler.metrics_reader_base import MetricsReaderBase
from smdebug.profiler.profiler_constants import (
    DEFAULT_PREFIX,
    ENV_TIME_BUFFER,
    HOROVODTIMELINE_SUFFIX,
    MODELTIMELINE_SUFFIX,
    PYTHONTIMELINE_SUFFIX,
    SMDATAPARALLELTIMELINE_SUFFIX,
    TENSORBOARDTIMELINE_SUFFIX,
    TIME_BUFFER_DEFAULT,
)
from smdebug.profiler.tf_profiler_parser import (
    HorovodProfilerEvents,
    SMDataParallelProfilerEvents,
    SMProfilerEvents,
    TensorboardProfilerEvents,
)
from smdebug.profiler.utils import (
    get_node_id_from_tracefilename,
    get_timestamp_from_tracefilename,
    is_valid_tfprof_tracefilename,
    is_valid_tracefilename,
)


class AlgorithmMetricsReader(MetricsReaderBase):
    # cache the fetched events in memory
    # if you have enough available memory subsequent fetch will be faster
    # if there is not enough memory, use use_in_memory_cache to use S3 or disk as cache
    def __init__(self, use_in_memory_cache=False):
        super().__init__(use_in_memory_cache)
        self.prefix = "framework"
        self._SMEventsParser = SMProfilerEvents()
        self._PythontimelineEventsParser = SMProfilerEvents()
        self._DetailedframeworkEventsParser = SMProfilerEvents(type="DetailedframeworkMetrics")
        self._TBEventsParser = TensorboardProfilerEvents()
        self._HorovordEventsParser = HorovodProfilerEvents()
        self._SMdataparallelEventsParser = SMDataParallelProfilerEvents()
        self._event_parsers = [
            self._PythontimelineEventsParser,
            self._DetailedframeworkEventsParser,
            self._TBEventsParser,
            self._HorovordEventsParser,
            self._SMdataparallelEventsParser,
        ]

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

        if end_time_microseconds >= self.get_timestamp_of_latest_available_file() or end_time_microseconds >= get_timestamp_from_tracefilename(
            self._startAfter_prefix
        ):
            self.refresh_event_file_list()

        timestamps = sorted(self._timestamp_to_filename.keys())

        # Find the timestamp that is greater than or equal start_time_microseconds. The tracefile corresponding to
        # that timestamp will contain events that are active during start_time_microseconds
        lower_bound_timestamp_index = bisect.bisect_left(timestamps, start_time_microseconds)

        # Find the timestamp that is immediate right to the end_time_microseconds. The tracefile corresponding to
        # that timestamp will contain events that are active during end_time_microseconds.
        upper_bound_timestamp_index = bisect.bisect_left(timestamps, end_time_microseconds)

        event_files = list()
        for index in timestamps[lower_bound_timestamp_index : upper_bound_timestamp_index + 1]:
            event_files.extend(self._timestamp_to_filename[index])
        self.logger.debug(f"event files to be fetched:{event_files}")
        return event_files

    """
    The function returns the right event parser for given file name
    1. For Filename containing 'pythontimeline.json'  -> SMEventsParser
    2. For Filename containing 'model_timeline.json'  -> SMEventsParser
    3. For Filename containing 'tensorboard' (TBD) -> TensorboardProfilerEvents
    4. For Filename containing 'horovod_timeline.json' -> 'HorovodProfilerEvents
    5. For Filename containing 'smdataparallel_timeline.json' -> 'SMDataParallelProfilerEvents
    """

    def _get_event_parser(self, filename):
        if PYTHONTIMELINE_SUFFIX in filename:
            return self._PythontimelineEventsParser
        if MODELTIMELINE_SUFFIX in filename:
            return self._DetailedframeworkEventsParser
        if TENSORBOARDTIMELINE_SUFFIX in filename:
            return self._TBEventsParser
        if HOROVODTIMELINE_SUFFIX in filename:
            return self._HorovordEventsParser
        if SMDATAPARALLELTIMELINE_SUFFIX in filename:
            return self._SMdataparallelEventsParser

    def _get_timestamp_from_filename(self, event_file):
        return get_timestamp_from_tracefilename(event_file)

    def _get_event_file_regex(self):
        return r"(.+)\.(json|csv|json.gz)$"


class LocalAlgorithmMetricsReader(AlgorithmMetricsReader):
    """
    The metrics reader is created with root folder in which the tracefiles are stored.
    """

    def __init__(self, trace_root_folder, use_in_memory_cache=False):
        self.trace_root_folder = trace_root_folder
        super().__init__(use_in_memory_cache)
        # Pre-build the file list so that user can query get_timestamp_of_latest_available_file() and get_current_time_range_for_event_query
        self.refresh_event_file_list()

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        self.logger.debug(f"Refreshing framework metrics from {self.trace_root_folder}")
        self._refresh_event_file_list_local_mode(self.trace_root_folder)

    def parse_event_files(self, event_files):
        self._parse_event_files_local_mode(event_files)


class S3AlgorithmMetricsReader(AlgorithmMetricsReader):
    """
    The s3_trial_path points to a s3 folder in which the tracefiles are stored. e.g.
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
            self.logger.info(
                f"S3AlgorithmMetricsReader created with bucket:{bucket_name} and prefix:{self.prefix}"
            )
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
        event_files_to_read = []

        for event_file in event_files:
            if event_file not in self._parsed_files:
                self.logger.debug(f"Will request s3 object {event_file}")
                event_files_to_read.append(event_file)
                file_read_requests.append(ReadObjectRequest(path=event_file))

        event_data_list = S3Handler.get_objects(file_read_requests)
        self.logger.debug(f"Got results back from s3 for {event_files}")
        for event_data, event_file in zip(event_data_list, event_files_to_read):
            self.logger.debug(f"Will parse events in event file:{event_file}")
            if event_file.endswith("json.gz") and is_valid_tfprof_tracefilename(event_file):
                self._get_event_parser(event_file).read_events_from_file(event_file)
                self._parsed_files.add(event_file)
            else:
                if is_valid_tracefilename(event_file):
                    event_string = event_data.decode("utf-8")
                    json_data = json.loads(event_string)
                    node_id = get_node_id_from_tracefilename(event_file)
                    self._get_event_parser(event_file).read_events_from_json_data(
                        json_data, node_id
                    )
                    self._parsed_files.add(event_file)
                else:
                    self.logger.info(f"Invalid tracefilename:{event_file} . Skipping.")

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        start_after = self._startAfter_prefix if self._startAfter_prefix else self.prefix
        self.logger.debug(
            f"Making listreq with bucket:{self.bucket_name} prefix:{self.prefix} startAfter:{start_after}"
        )
        list_dir = ListRequest(Bucket=self.bucket_name, Prefix=self.prefix, StartAfter=start_after)
        self._refresh_event_file_list_s3_mode(list_dir)
