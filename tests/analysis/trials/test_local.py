import uuid
import os
from tests.analysis.utils import generate_data, check_trial
from tornasole.trials.trial_catalog import LocalTrialCatalog
from tornasole.trials import LocalTrial

def check_local(localdir, trial_name, num_steps, num_tensors):
    tc = LocalTrialCatalog(localdir=localdir)
    assert trial_name in tc.list_candidates()
    path = os.path.join(localdir, trial_name)
    trial_obj = LocalTrial(name=trial_name, dirname=path)
    tc.add_trial(trial_name, trial_obj)
    trial_obj2 = tc.get_trial(trial_name)
    assert trial_obj == trial_obj2
    check_trial(trial_obj, num_tensors=num_tensors, num_steps=num_steps)

def test_local():
    trial_name = str(uuid.uuid4())
    path = 'ts_output/train/'
    num_steps = 20
    num_tensors = 10
    for i in range(num_steps):
        generate_data(path=path, trial=trial_name, num_tensors=10,
                      step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), rank=0)
    check_local(path, trial_name, num_steps=num_steps, num_tensors=num_tensors)
