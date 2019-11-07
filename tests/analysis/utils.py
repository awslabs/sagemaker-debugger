from tornasole.core.writer import FileWriter
import numpy as np
from tornasole.core.collection_manager import CollectionManager
from tornasole.core.config_constants import TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME
import os
import aioboto3
import asyncio
from tornasole.core.access_layer.s3handler import S3Handler, ListRequest


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
        c.export(os.path.join(path, trial), TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME)


def check_trial(trial_obj, num_steps, num_tensors):
    assert len(trial_obj.tensors()) == num_tensors
    for t in trial_obj.tensors():
        assert len(trial_obj.tensor(t).steps()) == num_steps
        for s in trial_obj.tensor(t).steps():
            v = trial_obj.tensor(t).value(s)
            assert v is not None


async def del_prefix_helper(bucket, keys):
    loop = asyncio.get_event_loop()
    client = aioboto3.client("s3", loop=loop)
    await asyncio.gather(*[client.delete_object(Bucket=bucket, Key=key) for key in keys])
    await client.close()


def delete_s3_prefix(bucket, prefix):
    s3_handler = S3Handler()
    list_req = [ListRequest(Bucket=bucket, Prefix=prefix)]
    keys = s3_handler.list_prefixes(list_req)[0]

    loop = asyncio.get_event_loop()
    task = loop.create_task(del_prefix_helper(bucket, keys))
    loop.run_until_complete(task)
