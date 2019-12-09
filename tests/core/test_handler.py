# Standard Library
import uuid

# Third Party
import pytest

# First Party
from smdebug.core.access_layer import TSAccessS3
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler


@pytest.mark.slow
def test_download_objects():
    s = uuid.uuid4()
    prefix = "test_get_objects/" + str(s)
    f = TSAccessS3("smdebugcodebuildtest", prefix, binary=False)
    f.write("a" * 100)
    f.write("b" * 200)
    f.write("c" * 300)
    f.close()
    r1 = ReadObjectRequest("s3://smdebugcodebuildtest/" + prefix)
    r2 = ReadObjectRequest("s3://smdebugcodebuildtest/" + prefix, start=100)
    r3 = ReadObjectRequest("s3://smdebugcodebuildtest/" + prefix, start=100, length=200)
    objects = S3Handler.get_objects([r1, r2, r3])
    assert objects[0].decode("ascii") == "a" * 100 + "b" * 200 + "c" * 300
    assert objects[1].decode("ascii") == "b" * 200 + "c" * 300, len(objects[1].decode("ascii"))
    assert objects[2].decode("ascii") == "b" * 200

    S3Handler.delete_prefix(path="s3://smdebugcodebuildtest/" + prefix)


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
    req1 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix)
    req2 = ListRequest(Bucket="smdebugcodebuildtest", Prefix="test_list_objects/", Delimiter="/")
    req3 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix, StartAfter=prefix + "/0")
    req4 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix, StartAfter=prefix + "/03")
    req5 = ListRequest(Bucket="smdebugcodebuildtest", Prefix=prefix + "/0")
    files = S3Handler.list_prefixes([req1, req2, req3, req4, req5])
    # test StartAfter and delimiters
    assert len(files[0]) == 4
    assert prefix + "/" in files[1]
    assert len(files[2]) == 4
    assert len(files[3]) == 2
    assert len(files[4]) == 3
    S3Handler.delete_prefix(path="s3://smdebugcodebuildtest/" + prefix)


@pytest.mark.slow
def test_delete_prefix():
    s = uuid.uuid4()
    prefix = "test_delete_prefix/" + str(s)
    for i in range(3):
        f = TSAccessS3("smdebugcodebuildtest", prefix + "/" + str(i))
        f.write(b"a")
        f.close()
    S3Handler.delete_prefix(path="s3://smdebugcodebuildtest/" + prefix)
    entries = S3Handler.list_prefix(ListRequest("smdebugcodebuildtest", prefix))
    assert len(entries) == 0


def check_performance():
    import time
    import multiprocessing

    kb = 1024
    mb = 1024 * 1024
    sizes = [10 * kb, 100 * kb, 500 * kb]  # , mb, 5 * mb, 10 * mb]
    num_files = [100, 1000, 10000]  # , 10000]  # , 100000]  # , 1000000]
    files_path = "smdebug-testing/resources/test_performance"
    times = []
    print("Size\tNumFiles\tPool size\tSync with multiprocessing")
    pool_sizes = [
        2 * multiprocessing.cpu_count(),
        4 * multiprocessing.cpu_count(),
        8 * multiprocessing.cpu_count(),
    ]
    for size in sizes:
        timesrow = []
        for nf in num_files:
            timesrow_for_pools = []
            for pool_size in pool_sizes:
                j = 0
                S3Handler.MULTIPROCESSING_POOL_SIZE = pool_size
                times_to_be_averaged = []
                reqs = [ReadObjectRequest(f"s3://{files_path}/{size}/{i}.dummy") for i in range(nf)]
                while j < 10:
                    sync_start = time.time()
                    S3Handler.get_objects(reqs, use_multiprocessing=True)
                    sync_end = time.time()
                    times_to_be_averaged.append(sync_end - sync_start)
                    j += 1
                timesrow_for_pools.append(
                    round(sum(times_to_be_averaged) / len(times_to_be_averaged), 2)
                )
            timesrow.append(timesrow_for_pools)
            print(f"{size} {nf} {pool_sizes} {timesrow_for_pools}")
        times.append(timesrow)
        print(f"Finished testing for {size}", times[-1])


if __name__ == "__main__":
    check_performance()
