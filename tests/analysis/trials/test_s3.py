from tornasole.trials import S3Trial
from tornasole.core.collection_manager import CollectionManager
from tornasole.core.config_constants import TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME
import uuid
import os
import pytest
from tornasole.core.utils import is_s3
from tests.analysis.utils import generate_data, check_trial, delete_s3_prefix


def check_s3_trial(path, num_steps=20, num_tensors=10):
    _, bucket, prefix = is_s3(path)
    trial_obj = S3Trial(name=prefix, bucket_name=bucket, prefix_name=prefix)
    check_trial(trial_obj, num_steps=num_steps, num_tensors=num_tensors)


@pytest.mark.slow
def test_s3():
    trial_name = str(uuid.uuid4())
    bucket = "tornasole-testing"
    path = "s3://" + os.path.join(bucket, "tornasole_outputs/")
    num_steps = 20
    num_tensors = 10
    for i in range(num_steps):
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=10,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
            rank=0,
        )
    check_s3_trial(os.path.join(path, trial_name), num_steps=num_steps, num_tensors=num_tensors)
    delete_s3_prefix("tornasole-testing", "tornasole_outputs/" + trial_name)


def help_test_multiple_trials(num_steps=20, num_tensors=10):
    trial_name = str(uuid.uuid4())
    bucket = "tornasole-testing"
    path = "s3://" + os.path.join(bucket, "tornasole_outputs/")

    c = CollectionManager()
    c.add("default")
    c.get("default").tensor_names = ["foo_" + str(i) for i in range(num_tensors)]
    c.export(path + trial_name, TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME)
    c.export(path + trial_name, TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME)
    for i in range(num_steps):
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
            rank=0,
        )
    _, bucket, prefix = is_s3(os.path.join(path, trial_name))
    trial_obj = S3Trial(name=prefix, bucket_name=bucket, prefix_name=prefix)
    return trial_obj, trial_name


@pytest.mark.slow
def test_multiple_s3_trials(num_trials=4, num_steps=5, num_tensors=5):
    data = [help_test_multiple_trials(num_steps, num_tensors) for i in range(num_trials)]
    trials = [d[0] for d in data]
    names = [d[1] for d in data]
    evals = [
        check_trial(trial_obj, num_steps=num_steps, num_tensors=num_tensors) for trial_obj in trials
    ]

    # delete the folders after the test
    for name in names:
        delete_s3_prefix("tornasole-testing", "tornasole_outputs/" + name)
