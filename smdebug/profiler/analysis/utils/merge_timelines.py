# Standard Library
import collections
import json
from enum import Enum

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.locations import TraceFileLocation
from smdebug.core.logger import get_logger
from smdebug.core.utils import is_s3
from smdebug.profiler.algorithm_metrics_reader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)

logger = get_logger()


class MergeUnit(Enum):
    """
    Enum to get Merge Unit - time or step.
    """

    # merge by time interval
    TIME = "time"

    # merge by step interval
    STEP = "step"


class MergedTimeline:
    def __init__(self, path, file_suffix_filter=None, output_directory=None):
        """
        :param path: trace root folder that contains framework and system folders
        :param file_suffix_filter: list of file suffix
            PYTHONTIMELINE_SUFFIX = "pythontimeline.json"
            MODELTIMELINE_SUFFIX = "model_timeline.json"
            TENSORBOARDTIMELINE_SUFFIX = "trace.json.gz"
            HOROVODTIMELINE_SUFFIX = "horovod_timeline.json".
            Default: None (all files will be merged)
        :param output_directory: Path where merged file should be saved
            Default: None (writes to the same location as the 'path' argument.
        """
        self.path = path
        self.out_dir = output_directory if output_directory is not None else self.path
        self.file_suffix_filter = file_suffix_filter

        self.bytes_written = 0

        # Reader for system and framework metrics
        if path.startswith("s3"):
            self.framework_metrics_reader = S3AlgorithmMetricsReader(self.path)
        else:
            self.framework_metrics_reader = LocalAlgorithmMetricsReader(self.path)

        self._writer = None
        self.tensor_table = collections.defaultdict(int)

    def open(self, file_path):
        """
        Open the trace event file
        """
        s3, bucket_name, key_name = is_s3(file_path)
        try:
            if s3:
                self._writer = TSAccessS3(bucket_name, key_name, binary=False)
            else:
                self._writer = TSAccessFile(file_path, "w")
        except (OSError, IOError) as err:
            logger.debug(f"Sagemaker-Debugger: failed to open {file_path}: {str(err)}")
        start, length = self._writer.write("[\n")
        self.bytes_written += length

    def file_name(self, end_timestamp_in_us):
        """
        Since this util will be used from a notebook or local directory, we directly write
        to the merged file
        """
        return TraceFileLocation().get_merged_trace_file_location(
            base_dir=self.out_dir, timestamp_in_us=end_timestamp_in_us
        )

    def merge_timeline(self, start, end, unit=MergeUnit.TIME):
        """
        Get all trace files captured and merge them for viewing in the browser
        """
        if unit == MergeUnit.STEP:
            start_timestamp_in_us, end_timestamp_in_us = self.framework_metrics_reader._get_time_interval_for_step(
                start, end
            )
        else:
            start_timestamp_in_us, end_timestamp_in_us = start, end

        filename = self.file_name(end_timestamp_in_us)
        if self._writer is None:
            self.open(filename)

        # get framework metrics
        self.framework_metrics_reader.refresh_event_file_list()
        framework_events = self.framework_metrics_reader.get_events(
            start_timestamp_in_us, end_timestamp_in_us, file_suffix_filter=self.file_suffix_filter
        )

        framework_events.sort(key=lambda x: x.start_time)

        for event in framework_events:
            if self.tensor_table[event.event_phase] == 0:
                tensor_idx = len(self.tensor_table)
                self.tensor_table[event.event_phase] = tensor_idx

                # Instant events don't have a training phase
                if event.phase != "i":
                    args = {"name": event.event_phase}
                    json_dict = {"name": "process_name", "ph": "M", "pid": tensor_idx, "args": args}
                    _, length = self._writer.write(json.dumps(json_dict) + ",\n")
                    self.bytes_written += length

                    args = {"sort_index": tensor_idx}
                    json_dict = {
                        "name": "process_sort_index",
                        "ph": "M",
                        "pid": tensor_idx,
                        "args": args,
                    }
                    _, length = self._writer.write(json.dumps(json_dict) + ",\n")
                    self.bytes_written += length

                if event.process_info is not None:
                    for thread in event.process_info._threads:
                        thread_info = event.process_info._threads[thread]
                        args = {"name": thread_info.thread_name}
                        json_dict = {
                            "name": "thread_name",
                            "ph": "M",
                            "pid": tensor_idx,
                            "tid": thread_info.tid,
                            "args": args,
                        }
                        _, length = self._writer.write(json.dumps(json_dict) + ",\n")
                        self.bytes_written += length

                        args = {"sort_index": thread_info.tid}
                        json_dict = {
                            "name": "thread_sort_index",
                            "ph": "M",
                            "pid": tensor_idx,
                            "tid": thread_info.tid,
                            "args": args,
                        }
                        _, length = self._writer.write(json.dumps(json_dict) + ",\n")
                        self.bytes_written += length

            event.pid = self.tensor_table[event.event_phase]
            _, length = self._writer.write(event.to_json() + ",\n")
            self.bytes_written += length

        self.close()

        get_logger("smdebug-profiler").info(f"Merged timeline saved at: {filename}")

        return filename

    def close(self):
        file_seek_pos = self.bytes_written - 2
        if isinstance(self._writer, TSAccessFile):
            self._writer._accessor.seek(file_seek_pos)
            self._writer._accessor.truncate()
        else:
            self._writer.data = self._writer.data[:file_seek_pos]

        if file_seek_pos > 2:
            self._writer.write("\n]")

        self._writer.flush()
        self._writer.close()
        self._writer = None
