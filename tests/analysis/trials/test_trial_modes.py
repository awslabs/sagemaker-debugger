import pytest
import uuid
import os
from tests.analysis.utils import generate_data
from tornasole.trials import LocalTrial, S3Trial

NUM_STEPS = 5
BUCKET = 'tornasole-testing'
SUB_BUCKET = 'tornasole_outputs/'
S3_PATH = 's3://' + os.path.join(BUCKET, SUB_BUCKET)


def test_local_index_mode():
    trial_name = str(uuid.uuid4())
    path = 'ts_output/train/'
    for i in range(NUM_STEPS):
        generate_data(path=path, trial=trial_name, num_tensors=10,
                      step=i, tname_prefix='foo',
                      worker='algo-1', shape=(3, 3, 3), rank=0)
        print("writing data to {}".format(path))
    trial_obj = LocalTrial(name=trial_name, dirname=os.path.join(path, trial_name))
    assert bool(trial_obj.index_tensors_dict) is True


@pytest.mark.slow # 0:03 to run
def test_local_event_mode():
    trial_name = str(uuid.uuid4())
    path = 'ts_output/train/'
    for i in range(NUM_STEPS):
        generate_data(path=path, trial=trial_name, num_tensors=10,
                      step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), rank=0)
    trial_obj = LocalTrial(name=trial_name, dirname=os.path.join(path, trial_name), index_mode=False)
    assert bool(trial_obj.index_tensors_dict) is False


@pytest.mark.slow # 0:12 to run
def test_s3_index_mode():
    trial_name = str(uuid.uuid4())
    for i in range(NUM_STEPS):
        generate_data(path=S3_PATH, trial=trial_name, num_tensors=10,
                      step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), rank=0)

    prefix_name = os.path.join('tornasole_outputs/', trial_name)
    trial_obj = S3Trial(name=trial_name, bucket_name=BUCKET, prefix_name=prefix_name)
    assert bool(trial_obj.index_tensors_dict) is True


@pytest.mark.slow # 0:12 to run
def test_s3_event_mode():
    trial_name = str(uuid.uuid4())
    for i in range(NUM_STEPS):
        generate_data(path=S3_PATH, trial=trial_name, num_tensors=10,
                      step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), rank=0)
    prefix_name = os.path.join(SUB_BUCKET, trial_name)
    trial_obj = S3Trial(name=trial_name, bucket_name=BUCKET, prefix_name=prefix_name, index_mode=False)
    assert bool(trial_obj.index_tensors_dict) is False
