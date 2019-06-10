import numpy as np
import pytest
import uuid
from tornasole_core.writer import FileWriter
from tornasole_core.reader import FileReader

def rw(path):
    """
    Checks that we can save data and read it back the way it was
    """
    with FileWriter(logdir=path, trial='my_trial', step=20, worker='algo-1') as fw:
        fname = fw.name()
        print( f'Saving data in {fname}')
        for i in range(10):
            data = np.ones(shape=(4,4), dtype=np.float32)*i
            fw.write_tensor(tdata=data, tname=f'foo_{i}')

    fr = FileReader(fname=fname)
    for i,ts in enumerate(fr.read_tensors(read_data=True)):
        """
        read_data returns name, step and data (if read_data==True)
        """
        print(i,ts)
        assert np.all(ts[2]==i)
    pass

#@pytest.mark.skip(reason="Local")
def test_local():
    rw('./ts_output/')

#@pytest.mark.skip(reason="No S3 client")
def test_s3():
    import boto3
    my_session = boto3.session.Session()
    my_region = my_session.region_name
    my_account = boto3.client('sts').get_caller_identity().get('Account')
    bucket_name = 'sagemaker-{}-{}'.format(my_region,my_account)
    key_name = 'tornasole/{}'.format(str(uuid.uuid4()))
    #sagemaker-us-east-1-722321484884
    location = 's3://{}/{}'.format(bucket_name,key_name)
    print("Saving to Location")
    rw(location)

#@pytest.mark.skip(reason="No string support")
def test_string():
    with FileWriter(logdir="./ts_output", trial='my_trial', step=20, worker='algo-1') as fw:
        fname = fw.name()
        print( f'Saving string data in {fname}')
        s_written = np.array(['foo', 'barz'])
        fw.write_tensor(tdata=s_written, tname=f'foo_string')

    fr = FileReader(fname=fname)
    read = list(fr.read_tensors(read_data=True))
    assert len(read)==1
    s_read = np.array(read[0][2])
    assert np.all(s_written == s_read)