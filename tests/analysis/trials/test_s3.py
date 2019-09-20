import aioboto3
import asyncio

from tornasole.core.access_layer.s3handler import *
from tornasole.trials import S3Trial
from tornasole.core.collection_manager import CollectionManager
import uuid
import os
from tornasole.core.utils import is_s3
from tests.analysis.utils import generate_data, check_trial

def check_s3_trial(path, num_steps=20, num_tensors=10):
    _, bucket, prefix = is_s3(path)
    trial_obj = S3Trial(name=prefix, bucket_name=bucket, prefix_name=prefix)
    check_trial(trial_obj, num_steps=num_steps, num_tensors=num_tensors)

async def del_folder(bucket, keys):
    loop = asyncio.get_event_loop()
    client = aioboto3.client('s3', loop=loop)
    await asyncio.gather(*[client.delete_object(Bucket=bucket, Key=key) for key in keys])
    await client.close()

def test_s3():
    trial_name = str(uuid.uuid4())
    bucket = 'tornasole-testing'
    path = 's3://' + os.path.join(bucket, 'tornasole_outputs/')
    num_steps = 20
    num_tensors = 10
    for i in range(num_steps):
        generate_data(path=path, trial=trial_name, num_tensors=10,
                      step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), rank=0)
    check_s3_trial(os.path.join(path, trial_name), num_steps=num_steps, num_tensors=num_tensors)

    # delete the bucket after the test
    s3_handler = S3Handler()
    list_req = [ListRequest(Bucket='tornasole-testing', Prefix="tornasole_outputs/" + trial_name)]
    keys = s3_handler.list_prefixes(list_req)[0]

    loop = asyncio.get_event_loop()
    task = loop.create_task(del_folder('tornasole-testing', keys))
    loop.run_until_complete(task)

def help_test_multiple_trials(num_steps = 20, num_tensors = 10):
    trial_name = str(uuid.uuid4())
    bucket = 'tornasole-testing'
    path = 's3://' + os.path.join(bucket, 'tornasole_outputs/')

    c = CollectionManager()
    c.add("default")
    c.get("default").tensor_names = ["foo_" + str(i) for i in range(num_tensors)]
    c.export(path + trial_name + "/collections.ts")
    c.export(path + trial_name + "/collections.ts")
    for i in range(num_steps):
        generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                      step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), rank=0)
    _, bucket, prefix = is_s3(os.path.join(path, trial_name))
    trial_obj = S3Trial(name=prefix, bucket_name=bucket, prefix_name=prefix)
    return trial_obj, trial_name

def test_multiple_s3_trials(num_trials = 4, num_steps = 5, num_tensors = 5):
    data = [help_test_multiple_trials(num_steps, num_tensors) for i in range(num_trials)]
    trials = [d[0] for d in data]
    names = [d[1] for d in data]
    evals = [check_trial(trial_obj, num_steps=num_steps, num_tensors=num_tensors) for trial_obj in trials]

    # delete the folders after the test
    for name in names:
        s3_handler = S3Handler()
        list_req = [ListRequest(Bucket='tornasole-testing', Prefix="tornasole_outputs/" + name)]
        keys = s3_handler.list_prefixes(list_req)[0]

        loop = asyncio.get_event_loop()
        task = loop.create_task(del_folder('tornasole-testing', keys))
        loop.run_until_complete(task)
