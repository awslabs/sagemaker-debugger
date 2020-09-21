# Standard Library
import uuid

# Third Party
import numpy as np
import pytest

# First Party
from smdebug.core.reader import FileReader
from smdebug.core.writer import FileWriter


def rw(path):
    """
    Checks that we can save data and read it back the way it was
    """
    with FileWriter(trial_dir=path + "/my_trial", step=20, worker="algo-1") as fw:
        for i in range(10):
            data = np.ones(shape=(4, 4), dtype=np.float32) * i
            fw.write_tensor(tdata=data, tname=f"foo_{i}")
        fname = fw.name()

    fr = FileReader(fname=fname)
    for i, ts in enumerate(fr.read_tensors()):
        """
        read_data returns name, step and data (if read_data==True)
        """
        print(i, ts)
        assert np.all(ts[2] == i)


# @pytest.mark.skip(reason="Local")
def test_local():
    rw("./ts_output/")


# @pytest.mark.skip(reason="No S3 client")
@pytest.mark.slow
def test_s3():
    import boto3

    my_session = boto3.session.Session()
    my_region = my_session.region_name
    my_account = boto3.client("sts").get_caller_identity().get("Account")
    bucket_name = "smdebug-testing"
    key_name = f"outputs/core-tests-{uuid.uuid4()}"
    # sagemaker-us-east-1-722321484884
    location = "s3://{}/{}".format(bucket_name, key_name)
    rw(location)


def test_string():
    with FileWriter(trial_dir="/tmp/ts_output/my_trial", step=20, worker="algo-1") as fw:
        s_written = np.array(["foo", "barz"])
        fw.write_tensor(tdata=s_written, tname=f"foo_string")
        fname = fw.name()
    fr = FileReader(fname=fname)
    read = list(fr.read_tensors())
    assert len(read) == 1
    s_read = np.array(read[0][2])
    assert np.all(s_written == s_read)
