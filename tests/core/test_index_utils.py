import pytest

from tornasole.core.utils import (
    is_s3,
    serialize_tf_device,
    deserialize_tf_device,
    parse_worker_name_from_file,
    get_worker_name_from_collection_file,
    get_path_to_collections,
)

from tornasole.core.s3_utils import list_s3_objects
from tornasole.core.index_reader import S3IndexReader


def test_tf_device_name_serialize_and_deserialize():
    import tensorflow as tf

    device_name = tf.test.gpu_device_name()
    if not bool(device_name):
        device_name = "/device:GPU:0"

    serialized_device_name = serialize_tf_device(device_name)
    assert deserialize_tf_device(serialized_device_name) == device_name

    device_name = "/replica:0/task:0/device:GPU:0"
    serialized_device_name = serialize_tf_device(device_name)
    assert deserialize_tf_device(serialized_device_name) == device_name


def test_parse_worker_name_from_index_file():
    filename = "/tmp/ts-logs/index/000000001/000000001230_worker_2.json"
    worker_name = parse_worker_name_from_file(filename)
    assert worker_name == "worker_2"

    filename = "/tmp/ts-logs/index/000000000499__job-worker_replica-0_task-1_device-GPU-6.json"
    worker_name = parse_worker_name_from_file(filename)
    assert worker_name == "/job:worker/replica:0/task:1/device:GPU:6"

    path = "s3://tornasole-testing/one-index-file"

    _, bucket, prefix = is_s3(path)

    index_files, _ = S3IndexReader.list_index_files(bucket, prefix)

    filename = index_files[0]
    worker_name = parse_worker_name_from_file(filename)
    assert worker_name == "/job:worker/replica:0/task:1/device:GPU:4"


def test_parse_worker_name_from_collection_file():
    path = "s3://tornasole-testing/one-index-file"
    _, bucket_name, key_name = is_s3(path)

    collection_files, _ = list_s3_objects(bucket_name, get_path_to_collections(key_name))

    assert len(collection_files) == 1

    collection_file = collection_files[0]
    worker_name = get_worker_name_from_collection_file(collection_file)
    assert worker_name == "/job:worker/replica:0/task:1/device:GPU:0"

    file_name = "/tmp/collections/000000000/job-worker_1_collections.json"
    worker_name = get_worker_name_from_collection_file(file_name)
    assert worker_name == "job-worker_1"
