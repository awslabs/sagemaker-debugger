# Standard Library
import calendar
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path

# Third Party
import pytest

# First Party
from smdebug.core.tfevent.timeline_file_writer import TimelineFileWriter
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    DEFAULT_PREFIX,
    TRACE_DIRECTORY_FORMAT,
)


@pytest.fixture()
def complete_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "complete_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture()
def file_open_fail_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "file_open_fail_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture()
def rotation_profiler_config_parser(config_folder, monkeypatch):
    def _choose_config(rotation_policy=None):
        nonlocal config_folder
        nonlocal monkeypatch
        config_path = os.path.join(
            config_folder, rotation_policy + "_rotation_profiler_config_parser.json"
        )
        monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
        return ProfilerConfigParser()

    return _choose_config


def test_create_timeline_file(simple_profiler_config_parser, out_dir):
    """
    This test is meant to test successful creation of the timeline file according to file path specification.
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYMMDDHR/$FILEEVENTSTARTTIMEUTCINEPOCH_
    {$ENV_NODE_ID_4digits0padded}_pythontimeline.json

    It reads backs the file contents to make sure it is in valid JSON format.
    """
    assert simple_profiler_config_parser.profiling_enabled

    timeline_writer = TimelineFileWriter(profiler_config_parser=simple_profiler_config_parser)
    assert timeline_writer

    for i in range(1, 11):
        n = "event" + str(i)
        timeline_writer.write_trace_events(
            training_phase="FileCreationTest", op_name=n, step_num=i, timestamp=time.time()
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    file_ts = files[0].name.split("_")[0]
    folder_name = files[0].parent.name
    assert folder_name == time.strftime(
        TRACE_DIRECTORY_FORMAT, time.gmtime(int(file_ts) / CONVERT_TO_MICROSECS)
    )
    assert folder_name == datetime.strptime(folder_name, TRACE_DIRECTORY_FORMAT).strftime(
        TRACE_DIRECTORY_FORMAT
    )

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict


def run(rank, profiler_config_parser):
    timeline_writer = TimelineFileWriter(profiler_config_parser=profiler_config_parser)
    assert timeline_writer

    for i in range(1, 6):
        n = "event" + str(i)
        timeline_writer.write_trace_events(
            training_phase="MultiProcessTest",
            op_name=n,
            step_num=0,
            worker=os.getpid(),
            process_rank=rank,
            timestamp=time.time(),
        )

    timeline_writer.flush()
    timeline_writer.close()


def test_multiprocess_write(simple_profiler_config_parser, out_dir):
    """
    This test is meant to test timeline events written multiple processes. Each process or worker, will have its own trace file.
    """
    assert simple_profiler_config_parser.profiling_enabled

    cpu_count = mp.cpu_count()

    processes = []
    for rank in range(cpu_count):
        p = mp.Process(target=run, args=(rank, simple_profiler_config_parser))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    assert len(files) == cpu_count

    event_ctr = 0
    for file_name in files:
        with open(file_name) as timeline_file:
            events_dict = json.load(timeline_file)
            for e in events_dict:
                if e["name"].startswith("event"):
                    event_ctr += 1

    assert event_ctr == cpu_count * 5


def test_duration_events(simple_profiler_config_parser, out_dir):
    """
    This test is meant to test duration events. By default, write_trace_events records complete events.
    TODO: Make TimelineWriter automatically calculate duration while recording "E" event
    """
    assert simple_profiler_config_parser.profiling_enabled

    timeline_writer = timeline_writer = TimelineFileWriter(
        profiler_config_parser=simple_profiler_config_parser
    )
    assert timeline_writer

    for i in range(1, 11):
        n = "event" + str(i)
        timeline_writer.write_trace_events(
            training_phase="DurationEventTest",
            op_name=n,
            step_num=i,
            phase="B",
            timestamp=time.time(),
        )
        timeline_writer.write_trace_events(
            training_phase="DurationEventTest",
            op_name=n,
            step_num=i,
            phase="E",
            timestamp=time.time(),
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict


@pytest.mark.slow
@pytest.mark.parametrize("policy", ["file_size", "file_interval"])
def test_rotation_policy(rotation_profiler_config_parser, policy, out_dir):
    """
    This test is meant to test if files are being closed and open correctly according to the 2 rotation policies -
    file_size -> close file if it exceeds certain size and open a new file
    file_interval -> close file if the file's folder was created before a certain time period and open a new file in a new folder
    :param policy: file_size or file_interval
    """
    rotation_profiler_config = rotation_profiler_config_parser(policy)
    assert rotation_profiler_config.profiling_enabled

    timeline_writer = TimelineFileWriter(profiler_config_parser=rotation_profiler_config)
    assert timeline_writer

    for i in range(1, 100):
        n = "event" + str(i)
        # adding a sleep here to trigger rotation policy
        time.sleep(0.05)
        timeline_writer.write_trace_events(
            training_phase=f"RotationPolicyTest_{policy}",
            op_name=n,
            step_num=i,
            timestamp=time.time(),
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    # check if files have been generated
    assert files

    # count the number of event JSON strings. This is to ensure all events have been written.
    # also check if the timestamp of all events in a file are <= filename timestamp
    event_ctr = 0
    start_time_since_epoch = 0
    file_details = []
    for file_name in files:
        if policy == "file_size":
            file_details.append(os.path.getsize(file_name))
        else:
            file_details.append(os.path.getmtime(file_name))
        path = file_name.name.split(DEFAULT_PREFIX)
        file_timestamp = int(path[0].split("_")[0])
        num_events_in_file = 0
        with open(file_name) as timeline_file:
            events_dict = json.load(timeline_file)
            for e in events_dict:
                if "args" in e and "start_time_since_epoch_in_micros" in e["args"]:
                    start_time_since_epoch = int(e["args"]["start_time_since_epoch_in_micros"])
                if "event" in e["name"]:
                    num_events_in_file += 1
                    event_ctr += 1
                    assert (
                        int(round(e["ts"] + start_time_since_epoch) / CONVERT_TO_MICROSECS)
                        <= file_timestamp
                    )
            # if rotation occurs too often, there might be only 1 event per file
            # the below assertion checks for this
            assert num_events_in_file >= 2

    if policy == "file_size":
        # assuming rotation max file size if 800 bytes, check if all the files are in the
        # range +- 60 bytes
        assert [pytest.approx(800, 60) == x for x in file_details]
    else:
        # assuming rotation file close interval is 0.5 seconds, check if the close time
        # difference between consecutive files is at least 0.5 seconds
        sorted(file_details)
        res = [j - i for i, j in zip(file_details[:-1], file_details[1:])]
        assert [x >= 0.5 for x in res]

    assert event_ctr == 99


@pytest.mark.parametrize("timezone", ["Europe/Dublin", "Australia/Melbourne", "US/Eastern"])
def test_utc_timestamp(monkeypatch, simple_profiler_config_parser, timezone, out_dir):
    """
    This test is meant to set to create files/events in different timezones and check if timeline writer stores
    them in UTC.
    """
    monkeypatch.setenv("TZ", timezone)
    assert simple_profiler_config_parser.profiling_enabled

    time.tzset()

    timeline_writer = TimelineFileWriter(profiler_config_parser=simple_profiler_config_parser)
    assert timeline_writer

    event_times_in_utc = []
    for i in range(1, 3):
        event_time_in_timezone = time.mktime(time.localtime())
        event_time_in_utc = time_in_utc = calendar.timegm(time.gmtime())
        event_times_in_utc.append(event_time_in_utc)
        timeline_writer.write_trace_events(
            training_phase=f"TimestampTest",
            op_name="event_in_" + timezone + str(i),
            timestamp=event_time_in_timezone,
            duration=20,
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    file_path = files[0]
    path = file_path.name.split(DEFAULT_PREFIX)
    file_timestamp = int(path[0].split("_")[0])

    # file timestamp uses end of last event
    assert (time_in_utc + 20) * CONVERT_TO_MICROSECS == file_timestamp

    start_time_since_epoch = 0
    idx = 0
    for file_name in files:
        with open(file_name) as timeline_file:
            events_dict = json.load(timeline_file)
            for e in events_dict:
                if "args" in e and "start_time_since_epoch_in_micros" in e["args"]:
                    start_time_since_epoch = int(e["args"]["start_time_since_epoch_in_micros"])
                if "event" in e["name"]:
                    assert (
                        e["ts"] + start_time_since_epoch
                        == event_times_in_utc[idx] * CONVERT_TO_MICROSECS
                    )
                    idx += 1


def test_file_open_fail(file_open_fail_profiler_config_parser):
    assert file_open_fail_profiler_config_parser.profiling_enabled

    # writing to an invalid path to trigger file open failure
    timeline_writer = TimelineFileWriter(
        profiler_config_parser=file_open_fail_profiler_config_parser
    )
    assert timeline_writer

    for i in range(1, 5):
        n = "event" + str(i)
        # Adding a sleep here to slow down event queuing
        time.sleep(0.001)
        timeline_writer.write_trace_events(
            training_phase=f"FileOpenTest", op_name=n, step_num=i, timestamp=time.time()
        )

    timeline_writer.flush()
    timeline_writer.close()

    # hacky way to check if the test passes
    assert not timeline_writer._worker._healthy


def test_events_far_apart(complete_profiler_config_parser, out_dir):
    assert complete_profiler_config_parser.profiling_enabled

    timeline_writer = TimelineFileWriter(profiler_config_parser=complete_profiler_config_parser)
    assert timeline_writer

    event_time_now = time.time()
    event_time_after_2hours = event_time_now + 120

    timeline_writer.write_trace_events(
        training_phase=f"FileOpenTest", op_name="event1", timestamp=event_time_now
    )
    time.sleep(2)
    timeline_writer.write_trace_events(
        training_phase=f"FileOpenTest", op_name="event2", timestamp=event_time_after_2hours
    )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    # rotate by file_size, gives 4 files - 1 per event
    # rotate by file_interval, gives 2 files
    assert len(files) == 2
