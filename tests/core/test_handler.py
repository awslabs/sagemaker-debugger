# Third Party
# Standard Library
import time
import uuid

import pytest

# First Party
from smdebug.core.access_layer import TSAccessS3
from smdebug.core.access_layer.s3handler import (
    ListRequest,
    ReadObjectRequest,
    S3Handler,
    S3HandlerAsync,
)


@pytest.mark.slow
def test_download_objects():
    s = uuid.uuid4()
    prefix = "test_get_objects/" + str(s)
    f = TSAccessS3("tornasolecodebuildtest", prefix, binary=False)
    f.write("a" * 100)
    f.write("b" * 200)
    f.write("c" * 300)
    f.close()
    handler = S3Handler()
    r1 = ReadObjectRequest("s3://tornasolecodebuildtest/" + prefix)
    r2 = ReadObjectRequest("s3://tornasolecodebuildtest/" + prefix, start=100)
    r3 = ReadObjectRequest("s3://tornasolecodebuildtest/" + prefix, start=100, length=200)
    objects = handler.get_objects([r1, r2, r3])
    assert objects[0].decode("ascii") == "a" * 100 + "b" * 200 + "c" * 300
    assert objects[1].decode("ascii") == "b" * 200 + "c" * 300, len(objects[1].decode("ascii"))
    assert objects[2].decode("ascii") == "b" * 200

    handler.delete_prefix(path="s3://tornasolecodebuildtest/" + prefix)


##########################################################
## Tests that listing of objects from S3 handler are working correctly
## Lists files from 4 different directories
## Also tests the StartAfter functionality and the delimiter and prefix functionality
@pytest.mark.slow
def test_list_objects():
    s = uuid.uuid4()
    prefix = "test_list_objects/" + str(s)
    for i in [0, 3, 7, 11]:
        f = TSAccessS3("tornasolecodebuildtest", prefix + "/" + format(i, "02"))
        f.write(b"a")
        f.close()
    handler = S3Handler()
    req1 = ListRequest(Bucket="tornasolecodebuildtest", Prefix=prefix)
    req2 = ListRequest(Bucket="tornasolecodebuildtest", Prefix="test_list_objects/", Delimiter="/")
    req3 = ListRequest(Bucket="tornasolecodebuildtest", Prefix=prefix, StartAfter=prefix + "/0")
    req4 = ListRequest(Bucket="tornasolecodebuildtest", Prefix=prefix, StartAfter=prefix + "/03")
    req5 = ListRequest(Bucket="tornasolecodebuildtest", Prefix=prefix + "/0")
    files = handler.list_prefixes([req1, req2, req3, req4, req5])
    # test StartAfter and delimiters
    assert len(files[0]) == 4
    assert prefix + "/" in files[1]
    assert len(files[2]) == 4
    assert len(files[3]) == 2
    assert len(files[4]) == 3
    handler.delete_prefix(path="s3://tornasolecodebuildtest/" + prefix)


@pytest.mark.slow
def test_delete_prefix():
    s = uuid.uuid4()
    prefix = "test_delete_prefix/" + str(s)
    for i in range(3):
        f = TSAccessS3("tornasolecodebuildtest", prefix + "/" + str(i))
        f.write(b"a")
        f.close()
    handler = S3Handler()
    handler.delete_prefix(path="s3://tornasolecodebuildtest/" + prefix)
    entries = handler.list_prefix(ListRequest("tornasolecodebuildtest", "test_delete_prefix"))
    assert len(entries) == 0


def performance_vs_async():
    kb = 1024
    mb = 1024 * 1024
    sizes = [10 * kb, 100 * kb, 500 * kb, mb, 5 * mb, 10 * mb]
    num_files = [100, 1000, 10000, 100000, 1000000]
    s = uuid.uuid4()
    prefix = "test_performance_prefix/" + str(s)
    # gen files
    def gen_data():
        for size in sizes:
            for i in range(num_files[-1]):
                f = TSAccessS3(
                    "tornasolecodebuildtest", f"{prefix}/{size}/{i}/.dummy", binary=False
                )
                f.write("a" * size)
                f.close()
            print(f"Generated data for {size}bytes")

    gen_data()
    handler = S3Handler()
    async_handler = S3HandlerAsync()

    times = []

    for size in sizes:
        timesrow = []
        for nf in num_files:
            reqs = []
            for i in range(nf):
                reqs.append(
                    ReadObjectRequest(f"s3://tornasolecodebuildtest/{prefix}/{size}/{i}.dummy")
                )
            sync_start = time.time()
            data1 = handler.get_objects(reqs)
            sync_end = time.time()

            async_start = time.time()
            data2 = async_handler.get_objects(reqs)
            async_end = time.time()

            assert data1 == data2
            print(f"Finished testing for {size} {nf}")
            timesrow.append((sync_end - sync_start, async_end - async_start))
        print(f"Finished testing for {size}")
        times.append(timesrow)


"""
# bash script to generate files for perf test
kb = 1024
mb = 1024 * 1024
sizes = [10 * kb, 100 * kb, 500 * kb, mb, 5 * mb, 10 * mb]
max_num_files = 100000

echo $sizes

"""
