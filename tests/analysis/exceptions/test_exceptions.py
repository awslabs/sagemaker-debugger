import shutil
import pytest
import uuid
from tests.analysis.utils import generate_data
from tornasole.trials import create_trial
from tornasole.exceptions import *
import boto3 as boto3
import os


def del_s3(bucket, file_path):
    s3_client = boto3.client("s3")
    s3_client.delete_object(Bucket=bucket, Key=file_path)


@pytest.mark.slow  # 0:40 to run
def test_refresh_tensors():
    trial_name = str(uuid.uuid4())
    path = "/tmp/tornasole_analysis_tests/test_refresh_tensors/"
    num_steps = 8
    num_tensors = 10
    for i in range(num_steps):
        if i % 2 == 0:
            continue
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
        )
    tr = create_trial(path + trial_name)
    assert len(tr.steps()) == 4

    try:
        tr.tensor("bar")
        assert False
    except TensorUnavailable:
        pass

    assert tr.tensor("foo_1") is not None
    assert tr.tensor("foo_1").value(num_steps - 1) is not None
    try:
        tr.tensor("foo_1").value(num_steps - 2)
        assert False
    except StepUnavailable:
        pass

    try:
        tr.tensor("foo_1").value(num_steps * 2)
        assert False
    except StepNotYetAvailable:
        pass

    for i in range(num_steps, num_steps * 2):
        if i % 2 == 0:
            continue
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
        )

    assert tr.tensor("foo_1").value(num_steps + 1) is not None
    try:
        tr.tensor("foo_1").value(num_steps)
        assert False
    except StepUnavailable:
        pass

    try:
        tr.tensor("foo_1").value(num_steps * 3)
        assert False
    except StepNotYetAvailable:
        pass

    shutil.rmtree(os.path.join(path, trial_name))
