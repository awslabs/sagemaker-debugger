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
    f = TSAccessS3("smdebugcodebuildtest", prefix, binary=False)
    f.write("a" * 100)
    f.write("b" * 200)
    f.write("c" * 300)
    f.close()
    handler = S3Handler()
    r1 = ReadObjectRequest("s3://smdebugcodebuildtest/" + prefix)
    r2 = ReadObjectRequest("s3://smdebugcodebuildtest/" + prefix, start=100)
    r3 = ReadObjectRequest("s3://smdebugcodebuildtest/" + prefix, start=100, length=200)
    objects = handler.get_objects([r1, r2, r3])
    assert objects[0].decode("ascii") == "a" * 100 + "b" * 200 + "c" * 300
    assert objects[1].decode("ascii") == "b" * 200 + "c" * 300, len(objects[1].decode("ascii"))
    assert objects[2].decode("ascii") == "b" * 200

    handler.delete_prefix(path="s3://smdebugcodebuildtest/" + prefix)


##########################################################
## Tests that listing of objects from S3 handler are working correctly
## Lists files from 4 different directories
## Also tests the StartAfter functionality and the delimiter and prefix functionality
@pytest.mark.slow
def test_list_objects():
    s = uuid.uuid4()
    prefix = "test_list_objects/" + str(s)
    for i in [0, 3, 7, 11]:
        f = TSAccessS3("smdebugcodebuildtest", prefix + "/" + format(i, "02"))
        f.write(b"a")
        f.close()
    handler = S3Handler()
    req1 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix)
    req2 = ListRequest(Bucket="smdebugcodebuildtest", Prefix="test_list_objects/", Delimiter="/")
    req3 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix, StartAfter=prefix + "/0")
    req4 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix, StartAfter=prefix + "/03")
    req5 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix + "/0")
    files = handler.list_prefixes([req1, req2, req3, req4, req5])
    # test StartAfter and delimiters
    assert len(files[0]) == 4
    assert prefix + "/" in files[1]
    assert len(files[2]) == 4
    assert len(files[3]) == 2
    assert len(files[4]) == 3
    handler.delete_prefix(path="s3://smdebugcodebuildtest/" + prefix)


@pytest.mark.slow
def test_delete_prefix():
    s = uuid.uuid4()
    prefix = "test_delete_prefix/" + str(s)
    for i in range(3):
        f = TSAccessS3("smdebugcodebuildtest", prefix + "/" + str(i))
        f.write(b"a")
        f.close()
    handler = S3Handler()
    handler.delete_prefix(path="s3://smdebugcodebuildtest/" + prefix)
    entries = handler.list_prefix(ListRequest("smdebugcodebuildtest", "test_delete_prefix"))
    assert len(entries) == 0


def performance_vs_async():
    kb = 1024
    mb = 1024 * 1024
    sizes = [10 * kb, 100 * kb, 500 * kb]  # , mb, 5 * mb, 10 * mb]
    num_files = [1, 10, 100, 1000, 3000, 10000, 100000]  # , 1000000]
    prefix = "test_performance_prefix"

    handler = S3Handler(use_s3_transfer=False)
    handler2 = S3Handler(use_s3_transfer=True)
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

            # sync2_start = time.time()
            # data2 = handler2.get_objects(reqs)
            # sync2_end = time.time()

            async_start = time.time()
            data3 = async_handler.get_objects(reqs)
            async_end = time.time()

            assert data1 == data3
            timesrow.append(
                (
                    sync_end - sync_start,
                    # sync2_end - sync2_start,
                    async_end - async_start,
                )
            )
            print(f"Finished testing for {size} {nf}", timesrow[-1])
        times.append(timesrow)
        print(f"Finished testing for {size}", times[-1])


if __name__ == "__main__":
    performance_vs_async()
"""
# bash commands to generate files for perf test
head -c 1048576 /dev/urandom > 0.dummy
for i in `seq 1 100000`; do aws s3 cp 0.dummy s3://smdebugcodebuildtest/test_performance_prefix/1048576/$i.dummy ; done;
"""
"""
sync, async
Finished testing for 10240 100  (5.466304063796997, 0.6316347122192383)
Finished testing for 10240 1000  (46.079941511154175, 4.562414169311523)
Finished testing for 10240 10000  (457.6105201244354, 59.778473138809204)

Finished testing for 10240 1 (0.0649409294128418, 0.09938454627990723)
Finished testing for 10240 10 (0.2014918327331543, 1.1134459972381592)
Finished testing for 10240 100 (2.155343532562256, 0.3696606159210205)
Finished testing for 10240 1000 (24.51427435874939, 3.7332723140716553)
---
without_s3_transfer, with_s3_transfer, async
Finished testing for 10240 1 (0.17870569229125977, 0.07202434539794922, 0.10649681091308594)
Finished testing for 10240 10 (0.1648552417755127, 0.29348111152648926, 0.07737946510314941)
Finished testing for 10240 100 (2.131474018096924, 3.889939308166504, 1.1508255004882812)
Finished testing for 10240 1000 (25.461544036865234, 49.814311027526855, 4.341756343841553)
"""
