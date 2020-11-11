# Standard Library

import bisect
import os
import re
import time

# First Party
from smdebug.core.access_layer.s3handler import S3Handler, is_s3
from smdebug.core.logger import get_logger
from smdebug.core.utils import list_files_in_directory
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    ENV_TRAIILING_DURATION,
    PYTHONTIMELINE_SUFFIX,
    TRAILING_DURATION_DEFAULT,
)
from smdebug.profiler.utils import (
    TimeUnits,
    convert_utc_timestamp_to_microseconds,
    is_valid_tfprof_tracefilename,
    is_valid_tracefilename,
    validate_system_profiler_file,
)


class MetricsReaderBase:
    def __init__(self, use_in_memory_cache=False):
        self.logger = get_logger()

        self._event_parsers = []

        # This is a list of timestamp -> [event_file] mapping
        self._timestamp_to_filename = dict()

        # This is a set of parsed event files. The entry is made into this file only if the complete file is read.
        self._parsed_files = set()

        # The startAfter_prefix is used in ListPrefix call to poll for available tracefiles in the S3 bucket. The
        # prefix lags behind the last polled tracefile by tunable trailing duration. This is to ensure that we do not
        # miss a
        # tracefile corresponding to timestamp earlier than last polled timestamp but arrived after we had polled.

        self._startAfter_prefix = ""
        self.prefix = ""
        self._cache_events_in_memory = use_in_memory_cache

    def get_all_event_parsers(self):
        return self._event_parsers

    """
    The function returns the timestamp of first available file.
    This timestamp indicates users can query the events up to this timestamp to gauge
    """

    def get_timestamp_of_first_available_file(self):
        return (
            sorted(self._timestamp_to_filename.keys())[0]
            if len(self._timestamp_to_filename) > 0
            else 0
        )

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
    This function queries the files that are currently available in the directory (for local mode) or in S3 for download.
    It rebuilds the map of timestamp to filename.
    """

    def refresh_event_file_list(self):
        pass

    def parse_event_files(self, event_files):
        pass

    def _get_event_parser(self, filename):
        pass

    def _get_event_file_regex(self):
        pass

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
        pass

    """
    The function returns the events that have recorded within the given time range.
    The function will download (or parse) the event files that are available
    for the given time range. It is possible that events are recorded during training but are not available for
    download.
    TODO: Implement blocking call to wait for files to be available for download.
    """

    def get_events(
        self,
        start_time,
        end_time,
        unit=TimeUnits.MICROSECONDS,
        event_type=None,
        file_suffix_filter=None,
    ):
        start_time = convert_utc_timestamp_to_microseconds(start_time, unit)
        end_time = convert_utc_timestamp_to_microseconds(end_time, unit)

        all_event_files = self._get_event_files_in_the_range(start_time, end_time)
        event_files = list()
        if file_suffix_filter is None:
            event_files = all_event_files
        else:
            for eventfile in all_event_files:
                if any([eventfile.endswith(suffix) for suffix in file_suffix_filter]):
                    event_files.append(eventfile)
        self.logger.info(f"Getting {len(event_files)} event files")
        self.logger.debug(f"Getting event files : {event_files} ")

        # Download files and parse the events
        self.parse_event_files(event_files)

        """
        We might have recorded events from different sources within this timerange.
        we will get the events from the relevant event parsers and merge them before returning.
        """
        result = []
        event_parsers = self.get_all_event_parsers()
        # Not all event parsers support event_type as input, only system_profiler_file_parser accepts it
        for eventParser in event_parsers:
            if event_type is not None:
                result.extend(
                    eventParser.get_events_within_time_range(
                        start_time, end_time, TimeUnits.MICROSECONDS, event_type
                    )
                )
            else:
                result.extend(
                    eventParser.get_events_within_time_range(
                        start_time, end_time, TimeUnits.MICROSECONDS
                    )
                )
            if not self._cache_events_in_memory:
                # clear eventParser events
                eventParser.clear_events()
                # cleanup parsed files set to force the reading of files again
                self._parsed_files = set()

        return result

    def _get_time_interval_for_step(self, start_step, end_step):
        """
        Use python timeline files to get time interval for a step interval
        """
        event_list = self.get_events(
            0,
            time.time() * CONVERT_TO_MICROSECS,
            TimeUnits.MICROSECONDS,
            file_suffix_filter=[PYTHONTIMELINE_SUFFIX],
        )

        start_time_us = end_time_us = None
        event_list.sort(key=lambda x: x.start_time)
        for event in event_list:
            if (
                hasattr(event, "event_args")
                and event.event_args is not None
                and "step_num" in event.event_args
            ):
                # get the start time of start step
                if start_time_us is None and start_step == int(event.event_args["step_num"]):
                    start_time_us = event.start_time
                # get the time just before the start of end_step
                if end_time_us is None and (end_step == int(event.event_args["step_num"])):
                    end_time_us = event.start_time - 1
            if start_time_us is not None and end_time_us is not None:
                break

        if start_time_us is None or end_time_us is None:
            get_logger().info(f"Invalid step interval [{start_step}, {end_step}]")
            start_time_us = end_time_us = 0
        return start_time_us, end_time_us

    def get_events_by_step(self, start_step, end_step, event_type=None, file_suffix_filter=None):
        """
        Get list of events by step interval
        """
        start_time, end_time = self._get_time_interval_for_step(start_step, end_step)

        return self.get_events(
            start_time, end_time, TimeUnits.MICROSECONDS, event_type, file_suffix_filter
        )

    """
    It is possible that event files from different nodes to arrive in S3 in different order. For example, Even if t1
    > t2, a event file with timestamp "t1" can arrive in S3 before the tracefile with timestamp "t2". If we list the
    prefix only on the basis of last arrived file (i.e. t1) we will miss the file for t2. Therefore, we will set the
    start prefix to a timestamp that is trailing behind the last timestamp by 'trailing duration'. This will ensure
    that we will attempt to get tracefiles with older timestamp even if they arrive late.
    """

    def _update_start_after_prefix(self):
        trailing_duration = os.getenv(ENV_TRAIILING_DURATION, TRAILING_DURATION_DEFAULT)
        sorted_timestamps = sorted(self._timestamp_to_filename.keys())
        if len(self._timestamp_to_filename) == 0:
            return
        last_timestamp_available = sorted_timestamps[-1]
        trailing_timestamp = last_timestamp_available - trailing_duration
        # Get the timestamp that is closely matching the trailing_timestamp
        trailing_timestamp = sorted_timestamps[
            bisect.bisect_left(sorted_timestamps, trailing_timestamp)
        ]
        self._startAfter_prefix = self._timestamp_to_filename[trailing_timestamp][0]
        s3, bucket_name, self._startAfter_prefix = is_s3(self._startAfter_prefix)

    """
    The function opens and reads the event files if they are not already parsed.
    For local metrics reader, we are currently assuming that the downloaded event file is a complete file.
    """

    def _parse_event_files_local_mode(self, event_files):
        for event_file in event_files:
            self.logger.debug(f"Going to parse file:{event_file}")
            if (
                is_valid_tracefilename(event_file)
                or is_valid_tfprof_tracefilename(event_file)
                or validate_system_profiler_file(event_file)
            ):
                if event_file not in self._parsed_files:
                    self._get_event_parser(event_file).read_events_from_file(event_file)
                    self._parsed_files.add(event_file)
            else:
                self.logger.info(
                    f"Skipping parsing of file {event_file} as this doesn't look like valid file"
                )

    def _get_timestamp_from_filename(self, event_file):
        pass

    """
    Create a map of timestamp to filename
    """

    def _refresh_event_file_list_s3_mode(self, list_dir):
        event_files = [
            x for x in S3Handler.list_prefix(list_dir) if re.search(self._get_event_file_regex(), x)
        ]
        for event_file in event_files:
            event_file_full_path = f"s3://{list_dir.bucket}/{event_file}"
            timestamp = self._get_timestamp_from_filename(event_file_full_path)
            if timestamp is None:
                self.logger.debug(f"Unable to find timestamp from event file name {event_file}.")
                continue
            if timestamp in self._timestamp_to_filename:
                if event_file_full_path not in self._timestamp_to_filename[timestamp]:
                    self._timestamp_to_filename[timestamp].append(event_file_full_path)
            else:
                self._timestamp_to_filename[timestamp] = [event_file_full_path]
        for timestamp in self._timestamp_to_filename:
            self._timestamp_to_filename[timestamp].sort()
        self._update_start_after_prefix()

    def _refresh_event_file_list_local_mode(self, trace_root_folder):
        path = os.path.expanduser(trace_root_folder)
        event_dir = os.path.join(path, self.prefix, "")
        event_files = list_files_in_directory(event_dir, file_regex=self._get_event_file_regex())
        for event_file in event_files:
            timestamp = self._get_timestamp_from_filename(event_file)
            if timestamp is None:
                self.logger.debug(f"Unable to find timestamp from event file name {event_file}.")
                continue
            if timestamp in self._timestamp_to_filename:
                if event_file not in self._timestamp_to_filename[timestamp]:
                    self._timestamp_to_filename[timestamp].append(event_file)
            else:
                self._timestamp_to_filename[timestamp] = [event_file]
        for timestamp in self._timestamp_to_filename:
            self._timestamp_to_filename[timestamp].sort()
