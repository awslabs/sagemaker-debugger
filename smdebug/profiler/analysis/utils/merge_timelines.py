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
            HOROVODTIMELINE_SUFFIX = "horovod_timeline.json"
            HERRINGTIMELINE_SUFFIX = "herring_timeline.json".
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
        # eventphase_starting_ids contains indexes for each of these phases,
        # The value of these is used to put pid for timeline. This makes sure that we will have timeline produced
        # which will be in order Step, Forward, Backward, DataIterator, Dataset, /device, /host, horovod, others..
        # Note that we can have 10000 event phase of each type
        self.eventphase_starting_ids = {
            "Step:": 1,
            "Forward": 10000,
            "Backward": 20000,
            "DataIterator": 30000,
            "Dataset": 40000,
            "/device": 50000,
            "/host": 60000,
            "gpu_functions": 70000,
            "cpu_functions": 80000,
            "Horovod": 90000,
            "other": 1000000,
        }

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
        print("Got events")
        framework_events.sort(key=lambda x: x.start_time)

        seen_phasepid_tids = {}
        print("Rewriting events")
        for event in framework_events:
            # print(str(event.tid) + "\n")
            if self.tensor_table[event.event_phase] == 0:
                # We will create tensor_idx based on what event_phase is there
                # tensor idx would be generated to show timeline in order of Step(0), Forward(1)/Backward(2), DataIterator(3), Dataset(4),
                # TF(/device, /host), PT detailed(cpu_functions/gpu_functions), horovod/herring
                # tensor_idx = len(self.tensor_table)

                found = False
                for key in self.eventphase_starting_ids.keys():
                    if key in event.event_phase:
                        tensor_idx = self.eventphase_starting_ids[key]
                        self.eventphase_starting_ids[key] += 1
                        found = True
                        break
                if not found:
                    tensor_idx = self.eventphase_starting_ids["other"]
                    self.eventphase_starting_ids["other"] += 1

                self.tensor_table[event.event_phase] = tensor_idx

                # Instant events don't have a training phase
                # TODO check with cycle marker
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

            tensor_idx = self.tensor_table[event.event_phase]
            # event.tid is not written. write it
            is_event_seen = False
            if (
                event.phase_pid in seen_phasepid_tids
                and event.tid in seen_phasepid_tids[event.phase_pid]
            ):
                is_event_seen = True

            if event.process_info is not None and not is_event_seen:
                phase_tid = event.phase_tid
                thread_info = event.process_info._threads[phase_tid]
                args = {"name": thread_info.thread_name}
                json_dict = {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": tensor_idx,
                    #
                    "tid": thread_info.tid,
                    "args": args,
                }
                _, length = self._writer.write(json.dumps(json_dict) + ",\n")
                self.bytes_written += length
                if event.phase_pid not in seen_phasepid_tids:
                    seen_phasepid_tids[event.phase_pid] = {}
                seen_phasepid_tids[event.phase_pid][event.tid] = 1
            # change event pid back tensor idx before writing it.
            event.pid = tensor_idx
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
