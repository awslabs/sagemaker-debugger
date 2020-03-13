# Standard Library
import json
import os
from pathlib import Path

# Third Party
import numpy as np

# First Party
from smdebug.core.access_layer.s3handler import DeleteRequest, S3Handler
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from smdebug.core.locations import IndexFileLocationUtils
from smdebug.core.writer import FileWriter


def generate_data(
    path,
    trial,
    step,
    tname_prefix,
    num_tensors,
    worker,
    shape,
    dtype=np.float32,
    rank=None,
    mode=None,
    mode_step=None,
    export_colls=True,
    data=None,
):
    with FileWriter(trial_dir=os.path.join(path, trial), step=step, worker=worker) as fw:
        for i in range(num_tensors):
            if data is None:
                data = np.ones(shape=shape, dtype=dtype) * step
            fw.write_tensor(tdata=data, tname=f"{tname_prefix}_{i}", mode=mode, mode_step=mode_step)
    if export_colls:
        c = CollectionManager()
        c.add("default")
        c.get("default").tensor_names = [f"{tname_prefix}_{i}" for i in range(num_tensors)]
        c.add("gradients")
        c.get("gradients").tensor_names = [f"{tname_prefix}_{i}" for i in range(num_tensors)]
        c.export(os.path.join(path, trial), DEFAULT_COLLECTIONS_FILE_NAME)


def check_trial(trial_obj, num_steps, num_tensors):
    assert len(trial_obj.tensor_names()) == num_tensors
    for t in trial_obj.tensor_names():
        assert len(trial_obj.tensor(t).steps()) == num_steps
        for s in trial_obj.tensor(t).steps():
            v = trial_obj.tensor(t).value(s)
            assert v is not None


def delete_s3_prefix(bucket, prefix):
    S3Handler.delete_prefix(delete_request=DeleteRequest(Bucket=bucket, Prefix=prefix))


def dummy_trial_creator(trial_dir, num_workers, job_ended):
    Path(trial_dir).mkdir(parents=True, exist_ok=True)
    cm = CollectionManager()
    for i in range(num_workers):
        collection_file_name = f"worker_{i}_collections.json"
        cm.export(trial_dir, collection_file_name)
    if job_ended:
        Path(os.path.join(trial_dir, "training_job_end.ts")).touch()


def dummy_step_creator(trial_dir, global_step, mode, mode_step, worker_name):
    static_step_data = (
        '{"meta": {"mode": "TRAIN", "mode_step": 0, "event_file_name": ""}, '
        '"tensor_payload": ['
        '{"tensorname": "gradients/dummy:0", "start_idx": 0, "length": 1}'
        "]}"
    )

    step = json.loads(static_step_data)
    step["meta"]["mode"] = mode
    step["meta"]["mode_step"] = mode_step

    index_file_location = IndexFileLocationUtils.get_index_key_for_step(
        trial_dir, global_step, worker_name
    )
    Path(os.path.dirname(index_file_location)).mkdir(parents=True, exist_ok=True)
    with open(index_file_location, "w") as f:
        json.dump(step, f)
