# First Party
from tornasole.trials import create_trial


def create_trial_fast_refresh(path, **kwargs):
    tr = create_trial(path, **kwargs)
    tr.training_end_delay_refresh = 0.01
    return tr
