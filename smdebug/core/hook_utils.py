# Standard Library
import os

# Local
from .access_layer.utils import check_dir_exists
from .logger import get_logger
from .utils import is_s3

logger = get_logger()


def verify_and_get_out_dir(out_dir):
    if out_dir is None:
        raise RuntimeError("out_dir is a required argument")
    is_s3_path, _, _ = is_s3(out_dir)
    if not is_s3_path:
        out_dir = os.path.expanduser(out_dir)
    # we check and raise error if directory already exists because
    # we don't want to merge tensors from current job with
    # tensors from previous job
    check_dir_exists(out_dir)
    return out_dir


def get_tensorboard_dir(export_tensorboard, tensorboard_dir, out_dir):
    if tensorboard_dir is not None:
        tensorboard_dir = os.path.expanduser(tensorboard_dir)

    if export_tensorboard and tensorboard_dir:
        return tensorboard_dir
    elif not export_tensorboard and tensorboard_dir:
        # Assume the user forgot `export_tensorboard` and save anyway.
        return tensorboard_dir
    elif export_tensorboard and not tensorboard_dir:
        return os.path.join(out_dir, "tensorboard")
    else:
        return None
