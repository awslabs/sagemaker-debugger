# Standard Library
import json
import os
import threading
import time

# First Party
from smdebug.core.tfevent.timeline_file_writer import TimelineFileWriter
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS, HOROVODTIMELINE_SUFFIX


"""
Horovod produces a large timeline file which is written to by all
Horovod workers. TraceFileRotation starts a reader thread that reads the
large Horovod file throughout the training process and splits the file into
smaller trace files based on Timeline File Writer's rotation policy.
"""


class HvdTraceFileRotation:
    def __init__(self, profiler_config_parser):
        self._profiler_config_parser = profiler_config_parser
        self.hvd_file = os.getenv("HOROVOD_TIMELINE", None)
        self.enabled = self._should_enable()
        if self.enabled:
            # base timestamp for event file
            self._base_timestamp_in_us = None

            # initial file seek position
            self.file_seek_pos = 0

            # thread event to break out of the thread loop
            self._stopper = threading.Event()

            # clock conversion from monotonic clock to std epoch time
            self.clock_conversion_in_us = int(
                round((time.monotonic() - time.time()) * CONVERT_TO_MICROSECS)
            )

            # default training phase name
            self.training_phase = {}

            # timeline writer for writing trace files
            self.tl_writer = TimelineFileWriter(
                profiler_config_parser=profiler_config_parser, suffix=HOROVODTIMELINE_SUFFIX
            )

            # Hvd file reader thread
            self._readerthread = threading.Thread(target=self._read_write_loop, daemon=True)
            self._readerthread.start()

    def _should_enable(self):
        """
        Enable Horovod file rotation if a timeline file will be written to (based
        on the env variable HOROVOD_TIMELINE) and if SM Profiler is enabled.
        """
        if self._profiler_config_parser.profiling_enabled and self.hvd_file:
            return True
        return False

    def _parse_trace_event(self, event):
        """
        Parse event to get some information that can be passed to timeline file writer
        """
        # all events that reach here must have timestamp

        # convert steady clock to system clock
        # This is the absolute timestamp
        timestamp_in_secs = (
            self._convert_monotonic_to_epoch_time(event["ts"]) / CONVERT_TO_MICROSECS
        )

        # make a note of duration if present
        duration_in_secs = event.get("dur", 0) / CONVERT_TO_MICROSECS

        # make a note of args if present
        args = event.get("args", {})

        # parse instant events which have a special field scope "s"
        if "s" in event:
            args.update({"s": event["s"]})

        # get the operation name from the event string
        op_name = event.get("name", "")
        # get the event pid
        pid = event.get("pid", 0)

        return op_name, timestamp_in_secs, duration_in_secs, pid, args

    def _convert_monotonic_to_epoch_time(self, timestamp_in_us):
        """
        Horovod writes events based on steady clock/monotonic clock.
        Convert this is standard clock which is time since epoch
        """
        return int(round(timestamp_in_us - self.clock_conversion_in_us))

    def _read_write_loop(self):
        """
        Reader thread to constantly read the large Horovod trace file and write
        events with Timeline File Writer
        """

        # Let the loop continuously run until stop event is set on smdebug hook cleanup
        while True:
            try:
                with open(self.hvd_file) as json_data:
                    # set the file pointer to the position up to which the reader
                    # thread has read.
                    json_data.seek(self.file_seek_pos)

                    # for every line read, verify that it is a valid JSON.
                    for line in json_data:
                        try:
                            event = (
                                json.loads(line[:-2])
                                if line.endswith(",\n")
                                else json.loads(line[:-1])
                            )

                            # the timestamp of the 1st event is considered as base timestamp
                            if self._base_timestamp_in_us is None:
                                if "ts" in event:
                                    timestamp = event["ts"]

                                    # find out the base timestamp
                                    # this is the base timestamp that will be used by timeline file writer as well.
                                    self._base_timestamp_in_us = self._convert_monotonic_to_epoch_time(
                                        timestamp
                                    )

                                    # Hvd base timestamp might be earlier than timeline writer's base start time.
                                    # Update timeline writer and the writer thread to avoid negative relative timestamp
                                    # in the rotated files.
                                    self.tl_writer._update_base_start_time(
                                        self._base_timestamp_in_us
                                    )

                            # the name mentioned in metadata events are used as training_phase in TimelineRecord
                            # make a note of this name. Timeline File Writer will take care of writing
                            # metadata event for each event
                            if event["ph"] == "M":
                                if "name" in event["args"]:
                                    self.training_phase[event["pid"]] = event["args"]["name"]
                            else:
                                # parse the event JSON string
                                op_name, timestamp_in_secs, duration, pid, args = self._parse_trace_event(
                                    event
                                )
                                # write complete, duration, and instant events
                                self.tl_writer.write_trace_events(
                                    training_phase=self.training_phase[pid],
                                    op_name=op_name,
                                    phase=event["ph"],
                                    timestamp=timestamp_in_secs,
                                    duration=duration,
                                    **args,
                                )
                        except ValueError:
                            # invalid JSON string, skip
                            pass
                    # update file seek position for the next read
                    self.file_seek_pos = max(self.file_seek_pos, json_data.tell())

                    # stop event has been set, exiting the thread
                    if self._stopper.isSet():
                        break
            except (OSError, IOError) as e:
                # unable to open timeline file, try again
                pass

            time.sleep(15)

    def close(self):
        """Flushes the trace event file to disk and close the file.
        """
        if self.enabled:
            # stop the reader thread
            self._stopper.set()
            self._readerthread.join()
            self.tl_writer.close()
