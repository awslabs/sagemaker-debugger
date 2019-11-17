# Standard Library
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


def parse_list_from_str(arg, delimiter=","):
    """
    :param arg: string or list of strings
    if it is string it is treated as character delimited string
    :param delimiter: string
    if arg is a string, this delimiter is used to split the string
    :return: list of strings
    """
    if arg is None:
        rval = []
    if isinstance(arg, str):
        if len(arg) == 0:
            rval = []
        else:
            rval = arg.split(delimiter)
    elif isinstance(arg, list):
        rval = arg
    return rval


def parse_bool(arg, default):
    if arg is None:
        return default
    elif arg in [False, True]:
        return arg
    elif arg in ["False", "false"]:
        return False
    elif arg in ["True", "true"]:
        return True
    else:
        raise ValueError("boolean argument expected, " "but found {}".format(arg))
