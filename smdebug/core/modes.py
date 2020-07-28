# Standard Library
from enum import Enum


# Note that Keras has similar concept of ModeKeys
class ModeKeys(Enum):
    TRAIN = 1  # training/fitting mode
    EVAL = 2  # testing/evaluation mode
    PREDICT = 3  # prediction/inference mode
    GLOBAL = 4


ALLOWED_MODES = [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT, ModeKeys.GLOBAL]
ALLOWED_MODE_NAMES = [x.name for x in ALLOWED_MODES]
MODE_STEP_PLUGIN_NAME = "mode_step"
MODE_PLUGIN_NAME = "mode"


def str_to_mode_keys(mode_str):
    if mode_str == "train":
        return ModeKeys.TRAIN
    elif mode_str == "eval":
        return ModeKeys.EVAL
    elif mode_str == "predict":
        return ModeKeys.PREDICT
    elif mode_str == "global":
        return ModeKeys.GLOBAL
    else:
        raise Exception("Invalid mode")
