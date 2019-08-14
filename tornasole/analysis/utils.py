from contextlib import contextmanager


@contextmanager
def no_refresh(trials):
    if isinstance(trials, list):
        for trial in trials:
            trial.dynamic_refresh = False
    else:
        trial = trials
        trial.dynamic_refresh = False

    yield trials

    if isinstance(trials, list):
        for trial in trials:
            trial.dynamic_refresh = True
    else:
        trial = trials
        trial.dynamic_refresh = True


@contextmanager
def refresh(trials):
    if isinstance(trials, list):
        for trial in trials:
            trial.dynamic_refresh = True
    else:
        trial = trials
        trial.dynamic_refresh = True

    yield trials

    if isinstance(trials, list):
        for trial in trials:
            trial.dynamic_refresh = False
    else:
        trial = trials
        trial.dynamic_refresh = False
