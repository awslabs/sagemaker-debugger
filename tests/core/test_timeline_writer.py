# Standard Library
import json
import multiprocessing as mp
import os

# First Party
from smdebug.core.config_constants import SM_PROFILER_FILE_PATH_ENV_STR
from smdebug.core.writer import FileWriter


def test_create_timeline_file(out_dir, monkeypatch):
    monkeypatch.setenv(SM_PROFILER_FILE_PATH_ENV_STR, out_dir + "/test_timeline.json")
    timeline_writer = FileWriter(trial_dir=out_dir, step=0, worker=str(os.getpid()), wtype="trace")
    assert timeline_writer

    timeline_writer.write_trace_events(tensor_name="FileCreationTest", op_name="event1", step_num=0)
    timeline_writer.write_trace_events(tensor_name="FileCreationTest", op_name="event2", step_num=1)

    timeline_writer.flush()
    timeline_writer.close()

    with open(out_dir + "/test_timeline.json") as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict


def run(rank, timeline_writer):
    timeline_writer.write_trace_events(
        tensor_name="MultiProcessTest",
        op_name="event1",
        step_num=0,
        worker=os.getpid(),
        process_rank=rank,
    )
    timeline_writer.write_trace_events(
        tensor_name="MultiProcessTest", op_name="event2", step_num=1, worker=os.getpid()
    )
    timeline_writer.flush()


def test_multiprocess_write(out_dir, monkeypatch):
    monkeypatch.setenv(SM_PROFILER_FILE_PATH_ENV_STR, out_dir + "/test_timeline.json")
    timeline_writer = FileWriter(trial_dir=out_dir, step=0, worker=str(os.getpid()), wtype="trace")
    assert timeline_writer

    cpu_count = mp.cpu_count()

    processes = []
    for rank in range(cpu_count):
        p = mp.Process(target=run, args=(rank, timeline_writer))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    timeline_writer.close()

    with open(out_dir + "/test_timeline.json") as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict
