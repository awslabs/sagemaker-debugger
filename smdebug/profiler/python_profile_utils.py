# Standard Library
import os
from enum import Enum, IntEnum

# First Party
from smdebug.profiler.profiler_constants import CPROFILE_NAME, PYINSTRUMENT_NAME


class PythonProfileModes(IntEnum):
    TRAIN = 1  # training/fitting mode
    EVAL = 2  # testing/evaluation mode
    PREDICT = 3  # prediction/inference mode
    GLOBAL = 4  # default mode
    PRE_STEP_ZERO = 5  # when hook is imported
    POST_HOOK_CLOSE = 6  # when hook is closed


class StepPhase(Enum):
    START = "start"  # pre-step zero
    STEP_START = "stepstart"  # start of step
    FORWARD_PASS_END = "forwardpassend"  # end of forward pass
    STEP_END = "stepend"  # end of training step
    END = "end"  # end of script


class PythonProfilerName(Enum):
    CPROFILE = CPROFILE_NAME
    PYINSTRUMENT = PYINSTRUMENT_NAME


class cProfileTimer(Enum):
    TOTAL_TIME = "total_time"
    CPU_TIME = "cpu_time"
    OFF_CPU_TIME = "off_cpu_time"
    DEFAULT = "default"


def str_to_python_profile_mode(mode_str):
    if mode_str == "train":
        return PythonProfileModes.TRAIN
    elif mode_str == "eval":
        return PythonProfileModes.EVAL
    elif mode_str == "predict":
        return PythonProfileModes.PREDICT
    elif mode_str == "global":
        return PythonProfileModes.GLOBAL
    elif mode_str == "prestepzero":
        return PythonProfileModes.PRE_STEP_ZERO
    elif mode_str == "posthookclose":
        return PythonProfileModes.POST_HOOK_CLOSE
    else:
        raise Exception("Invalid mode")


def python_profile_mode_to_str(mode):
    if mode == PythonProfileModes.TRAIN:
        return "train"
    elif mode == PythonProfileModes.EVAL:
        return "eval"
    elif mode == PythonProfileModes.PREDICT:
        return "predict"
    elif mode == PythonProfileModes.GLOBAL:
        return "global"
    elif mode == PythonProfileModes.PRE_STEP_ZERO:
        return "prestepzero"
    elif mode == PythonProfileModes.POST_HOOK_CLOSE:
        return "posthookclose"
    else:
        raise Exception("Invalid mode")


def mode_keys_to_python_profile_mode(mode):
    return PythonProfileModes(mode.value)


def total_time():
    if not os.times:
        return -1
    times = os.times()
    return times.elapsed


def off_cpu_time():
    if not os.times:
        return -1
    times = os.times()
    return times.elapsed - (times.system + times.user)


def cpu_time():
    if not os.times:
        return -1
    times = os.times()
    return times.system + times.user
