# Standard Library
import os
from pathlib import Path

# Third Party
import tensorflow as tf
from packaging import version

# First Party
from smdebug.profiler.profiler_constants import TENSORBOARDTIMELINE_SUFFIX
from smdebug.profiler.tf_profiler_parser import TensorboardProfilerEvents


def verify_detailed_profiling(out_dir, expected_event_count):
    """
    This verifies the number of events when detailed profiling is enabled.
    """
    t_events = TensorboardProfilerEvents()

    if version.parse(tf.__version__) <= version.parse("2.10"):
        # get tensorboard timeline files
        files = list(Path(os.path.join(out_dir, "framework")).rglob(f"*{TENSORBOARDTIMELINE_SUFFIX}"))

        assert len(files) == 1

        trace_file = str(files[0])
        t_events.read_events_from_file(trace_file)

        all_trace_events = t_events.get_all_events()
        num_trace_events = len(all_trace_events)

        print(f"Number of events read = {num_trace_events}")

        # The number of events is varying by a small number on
        # consecutive runs. Hence, the approximation in the below asserts.
        assert num_trace_events >= expected_event_count, f"{num_trace_events}"
