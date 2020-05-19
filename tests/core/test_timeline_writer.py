# Standard Library
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

# Third Party
import pytest
from tests.analysis.utils import delete_s3_prefix

# First Party
from smdebug.core.access_layer.s3handler import ReadObjectRequest, S3Handler
from smdebug.core.s3_utils import list_s3_objects
from smdebug.core.utils import load_json_as_dict
from smdebug.core.writer import FileWriter


def helper_get_trial_dir(trial_dir, file_path="local"):
    if file_path == "s3":
        trial_name = str(uuid.uuid4())
        bucket = "kannanva-smdebug-testing"
        prefix = "outputs/" + trial_name
        trial_dir = "s3://" + os.path.join(bucket, prefix)

    return trial_dir


@pytest.mark.parametrize("file_path", ["local", "s3"])
def test_create_timeline_file(out_dir, file_path):
    """
    This test is meant to test successful creation of the timeline file according to file path specification.
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYMMDDHR/$FILEEVENTSTARTTIMEUTCINEPOCH_
    {$ENV_NODE_ID_4digits0padded}_pythontimeline.json

    It reads backs the file contents to make sure it is in valid JSON format.
    """
    if file_path == "s3":
        trial_name = str(uuid.uuid4())
        bucket = "kannanva-smdebug-testing"
        prefix = "outputs/" + trial_name
        trial_dir = "s3://" + os.path.join(bucket, prefix)
    else:
        trial_dir = out_dir
    timeline_writer = FileWriter(
        trial_dir=trial_dir, step=0, worker=str(os.getpid()), wtype="trace"
    )
    assert timeline_writer

    for i in range(1, 11):
        n = "event" + str(i)
        timestamp = None
        # setting timestamp half the time
        if i % 2 == 0:
            timestamp = time.time()
        timeline_writer.write_trace_events(
            training_phase="FileCreationTest", op_name=n, step_num=i, timestamp=timestamp
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    if file_path == "s3":
        files, _ = list_s3_objects(bucket=bucket, prefix=prefix)
    else:
        for path in Path(trial_dir + "/framework/pevents").rglob("*.json"):
            files.append(path)

    assert len(files) == 1

    if file_path == "s3":
        request = ReadObjectRequest("s3://" + os.path.join(bucket, files[0]))
        obj_data = S3Handler.get_object(request)
        obj_data = obj_data.decode("utf-8")
        events_dict = load_json_as_dict(obj_data)
    else:
        with open(files[0]) as timeline_file:
            events_dict = json.load(timeline_file)

    assert events_dict

    if file_path == "s3":
        delete_s3_prefix(bucket=bucket, prefix=prefix)


def run(rank, out_dir):
    timeline_writer = FileWriter(trial_dir=out_dir, step=0, worker=str(os.getpid()), wtype="trace")
    assert timeline_writer

    for i in range(1, 6):
        n = "event" + str(i)
        timeline_writer.write_trace_events(
            training_phase="MultiProcessTest",
            op_name=n,
            step_num=0,
            worker=os.getpid(),
            process_rank=rank,
        )

    timeline_writer.flush()
    timeline_writer.close()


@pytest.mark.parametrize("file_path", ["local", "s3"])
def test_multiprocess_write(out_dir, file_path):
    """
    This test is meant to test timeline events written multiple processes. Each process or worker, will have its own trace file.
    """
    if file_path == "s3":
        trial_name = str(uuid.uuid4())
        bucket = "kannanva-smdebug-testing"
        prefix = "outputs/" + trial_name
        trial_dir = "s3://" + os.path.join(bucket, prefix)
    else:
        trial_dir = out_dir
    cpu_count = mp.cpu_count()

    processes = []
    for rank in range(cpu_count):
        p = mp.Process(target=run, args=(rank, trial_dir))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    files = []
    if file_path == "s3":
        files, _ = list_s3_objects(bucket=bucket, prefix=prefix)
    else:
        for path in Path(trial_dir + "/framework/pevents").rglob("*.json"):
            files.append(path)

    assert len(files) == cpu_count

    event_ctr = 0
    if file_path == "s3":
        for file_name in files:
            request = ReadObjectRequest("s3://" + os.path.join(bucket, file_name))
            obj_data = S3Handler.get_object(request)
            obj_data = obj_data.decode("utf-8")
            events_dict = load_json_as_dict(obj_data)
            for e in events_dict:
                if e["name"].startswith("event"):
                    event_ctr += 1
    else:
        for file_name in files:
            with open(file_name) as timeline_file:
                events_dict = json.load(timeline_file)
                for e in events_dict:
                    if e["name"].startswith("event"):
                        event_ctr += 1

    assert event_ctr == cpu_count * 5

    if file_path == "s3":
        delete_s3_prefix(bucket=bucket, prefix=prefix)


@pytest.mark.parametrize("file_path", ["local", "s3"])
def test_duration_events(out_dir, file_path):
    """
    This test is meant to test duration events. By default, write_trace_events records complete events.
    TODO: Make TimelineWriter automatically calculate duration while recording "E" event
    """
    if file_path == "s3":
        trial_name = str(uuid.uuid4())
        bucket = "kannanva-smdebug-testing"
        prefix = "outputs/" + trial_name
        trial_dir = "s3://" + os.path.join(bucket, prefix)
    else:
        trial_dir = out_dir
    timeline_writer = FileWriter(
        trial_dir=trial_dir, step=0, worker=str(os.getpid()), wtype="trace"
    )
    assert timeline_writer

    for i in range(1, 11):
        n = "event" + str(i)
        timeline_writer.write_trace_events(
            training_phase="DurationEventTest", op_name=n, step_num=i, phase="B"
        )
        timeline_writer.write_trace_events(
            training_phase="DurationEventTest", op_name=n, step_num=i, phase="E"
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    if file_path == "s3":
        files, _ = list_s3_objects(bucket=bucket, prefix=prefix)
    else:
        for path in Path(trial_dir + "/framework/pevents").rglob("*.json"):
            files.append(path)

    assert len(files) == 1

    if file_path == "s3":
        request = ReadObjectRequest("s3://" + os.path.join(bucket, files[0]))
        obj_data = S3Handler.get_object(request)
        obj_data = obj_data.decode("utf-8")
        events_dict = load_json_as_dict(obj_data)
    else:
        with open(files[0]) as timeline_file:
            events_dict = json.load(timeline_file)

    assert events_dict

    if file_path == "s3":
        delete_s3_prefix(bucket=bucket, prefix=prefix)


@pytest.mark.slow
@pytest.mark.parametrize("file_path", ["local", "s3"])
@pytest.mark.parametrize("policy", ["file_size", "file_interval"])
def test_rotation_policy(out_dir, monkeypatch, policy, file_path):
    """
    This test is meant to test if files are being closed and open correctly according to the 2 rotation policies -
    file_size -> close file if it exceeds certain size and open a new file
    file_interval -> close file if the file's folder was created before a certain time period and open a new file in a new folder
    :param policy: file_size or file_interval
    """
    if file_path == "s3":
        trial_name = str(uuid.uuid4())
        bucket = "kannanva-smdebug-testing"
        prefix = "outputs/" + trial_name
        trial_dir = "s3://" + os.path.join(bucket, prefix)
    else:
        trial_dir = out_dir
    if policy == "file_size":
        monkeypatch.setenv("ENV_MAX_FILE_SIZE", "300")  # rotate file if size > 300 bytes
    elif policy == "file_interval":
        monkeypatch.setenv(
            "ENV_CLOSE_FILE_INTERVAL", "1"
        )  # rotate file if file interval > 1 second

    timeline_writer = FileWriter(
        trial_dir=trial_dir, step=0, worker=str(os.getpid()), wtype="trace", flush_secs=1
    )
    assert timeline_writer

    for i in range(1, 5):
        n = "event" + str(i)
        # adding a sleep here to trigger rotation policy
        time.sleep(1)
        timeline_writer.write_trace_events(
            training_phase=f"RotationPolicyTest_{policy}", op_name=n, step_num=i
        )

    timeline_writer.flush()
    timeline_writer.close()

    files = []
    if file_path == "s3":
        files, _ = list_s3_objects(bucket=bucket, prefix=prefix)
    else:
        for path in Path(trial_dir + "/framework/pevents").rglob("*.json"):
            files.append(path)

    # rotate by file_size, gives 4 files - 1 per event
    # rotate by file_interval, gives 2 files
    assert len(files) >= 2

    # count the number of event JSON strings. This is to ensure all events have been written.
    event_ctr = 0
    if file_path == "s3":
        for file_name in files:
            request = ReadObjectRequest("s3://" + os.path.join(bucket, file_name))
            obj_data = S3Handler.get_object(request)
            obj_data = obj_data.decode("utf-8")
            events_dict = load_json_as_dict(obj_data)
            for e in events_dict:
                if e["name"].startswith("event"):
                    event_ctr += 1
    else:
        for file_name in files:
            with open(file_name) as timeline_file:
                events_dict = json.load(timeline_file)
                for e in events_dict:
                    if e["name"].startswith("event"):
                        event_ctr += 1

    assert event_ctr == 4

    if file_path == "s3":
        delete_s3_prefix(bucket=bucket, prefix=prefix)
