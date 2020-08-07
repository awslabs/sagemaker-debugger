# Standard Library
import glob
import gzip
import json
import os
from datetime import datetime

# First Party
from smdebug.core.logger import get_logger
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    TENSORBOARDTIMELINE_SUFFIX,
    TF_METRICS_PREFIX,
)
from smdebug.profiler.trace_event_file_parser import TraceEventParser
from smdebug.profiler.utils import read_tf_profiler_metadata_file


class SMProfilerEvents(TraceEventParser):
    def __init__(self, type="PythontimelineMetrics"):
        super().__init__()
        self.type = type

    def type(self):
        return self.type


class TensorboardProfilerEvents(TraceEventParser):
    def __init__(self):
        super().__init__()

    def type(self):
        return "TFProfilerMetrics"

    def _get_trace_events_json(self, tracefile):
        try:
            with gzip.GzipFile(tracefile, "r") as fin:
                trace_json_data = json.loads(fin.read().decode("utf-8"))
        except Exception as e:
            self.logger.error(f"Can't open TF trace file {tracefile}: Exception {str(e)} ")
            return None
        if "traceEvents" not in trace_json_data:
            self.logger.error(f"The TF trace file {tracefile} does not contain traceEvents")
            return None
        trace_events_json = trace_json_data["traceEvents"]
        _, start_time_in_micros, _ = read_tf_profiler_metadata_file(tracefile)
        # the first time profiler.start() is called is considered the start time
        # for TF profiler
        metadata = []
        args = {"start_time_since_epoch_in_micros": int(start_time_in_micros)}
        json_dict = {"name": "process_name", "ph": "M", "pid": 0, "args": args}
        metadata.append(json_dict)
        args = {"sort_index": 0}
        json_dict = {"name": "process_sort_index", "ph": "M", "pid": 0, "args": args}
        metadata.append(json_dict)

        # insert metadata at the beginning of trace events json
        trace_events_json = metadata + trace_events_json
        return trace_events_json

    def read_events_from_file(self, tracefile):
        trace_events_json = self._get_trace_events_json(tracefile)

        if trace_events_json:
            for event in trace_events_json:
                self._read_event(event)

    def get_events_within_range(self, start_time: datetime, end_time: datetime):
        return None

    def _get_event_phase(self, event):
        if not event.event_name or not event.event_name.startswith(TF_METRICS_PREFIX):
            return

        # Phase is between aws-marker and first slash.
        phase = event.event_name.split("/")[0][len(TF_METRICS_PREFIX) :]

        if phase in ["ForwardPass", "ComputeGradient", "ApplyGradient"]:
            return phase

    def get_complete_op_events(self, tracefile):
        op_events = []
        self.read_events_from_file(tracefile)
        all_events = self.get_all_events()
        for event in all_events:
            if event.event_args is not None:
                phase = self._get_event_phase(event)
                if phase:
                    op_events.append((event, phase))
        return op_events

    def get_training_info(self, tracefile):
        all_op_events = self.get_complete_op_events(tracefile)
        # each in the list , will be list [name, ts, duration]
        training_annotation = {
            "ForwardPass": [],
            # "BackwardPass": [],
            "ComputeGradient": [],
            "ApplyGradient": [],
        }

        for event, phase in all_op_events:
            training_annotation[phase].append([event.event_name, event.start_time, event.duration])
        return training_annotation

    # TODO: The following function has to be revisited when
    # AWS-TF forward/backward annotations are finalized.
    def _dump_info_to_json(self, training_info, trace_json_file):
        """
        This function dumps the training info gathered into the
        json file passed.
        """
        with open(trace_json_file, "r+") as f:
            data = json.load(f)
        f.close()

        for phase, metrics in training_info.items():
            if not metrics:
                get_logger("smdebug-profiler").error(
                    f"No metrics captured after profiling for {phase}!"
                )
                continue

            # Getting the min start_time to get the start_time
            start = min(x[1] for x in metrics)
            # Calculating the max end time using duration.
            end = max(x[1] + x[2] for x in metrics)
            phase = "BackwardPass" if phase != "ForwardPass" else phase
            main_entry = {
                "pid": "/" + phase,
                "tid": phase,
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "name": phase,
                "args": {"group_id": phase, "long_name": phase},
            }
            data["traceEvents"].append(main_entry)

            for idx, metrics in enumerate(metrics):
                entry = {
                    "pid": "/" + phase,
                    "tid": phase + "ops",
                    "ph": "X",
                    "args": {"group_id": phase, "long_name": metrics[0]},
                    "ts": metrics[1] / 1000,
                    "dur": metrics[2] / 1000,
                    "name": metrics[0],
                }
                data["traceEvents"].append(entry)

        get_logger("smdebug-profiler").info(f"Dumping into file {trace_json_file}")
        with open(trace_json_file, "w+") as outfile:
            json.dump(data, outfile)

    # TODO: The following function has to be revisited when
    # AWS-TF forward/backward annotations are finalized.
    def parse_tf_native_profiler_trace_json(self, log_dir):
        """
        Returns: Function returns a dictonary of
                {"ForwardPass": [],  "ComputeGradient": [], "ApplyGradient": []}
                The value is list of list. Each list is [opname, ts, duration]

        """
        tf_profiler_folders = os.listdir(os.path.join(log_dir + "/plugins/profile"))
        trace_json_files = []
        for folder in tf_profiler_folders:
            folderpath = os.path.join(log_dir + "/plugins/profile", folder)
            for file in os.listdir(folderpath):
                if file.endswith(".gz"):
                    trace_json_files.append(os.path.join(folderpath, file))

        # get the latest file. TF annotations will be appended to this file
        trace_json_file = max(glob.glob(trace_json_files), key=os.path.getmtime)
        training_info = self.get_training_info(trace_json_file)

        # Dump gathered data into trace_json_file
        self._dump_info_to_json(training_info, trace_json_file)

        return training_info, trace_json_file


class HorovodProfilerEvents(TraceEventParser):
    def __init__(self):
        super().__init__()

    def type(self):
        return "HorovodMetrics"
