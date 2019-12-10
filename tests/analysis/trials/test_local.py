# Standard Library
import os
import uuid

# Third Party
from tests.analysis.utils import check_trial, generate_data

# First Party
from smdebug.trials import LocalTrial


def check_local(localdir, trial_name, num_steps, num_tensors):
    path = os.path.join(localdir, trial_name)
    trial_obj = LocalTrial(name=trial_name, dirname=path)
    check_trial(trial_obj, num_tensors=num_tensors, num_steps=num_steps)


def test_local():
    trial_name = str(uuid.uuid4())
    path = "ts_output/train/"
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
    check_local(path, trial_name, num_steps=num_steps, num_tensors=num_tensors)
