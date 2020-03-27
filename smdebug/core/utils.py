# Standard Library
import bisect
import json
import os
import re
import shutil
import socket
from pathlib import Path
from typing import Dict, List

# First Party
from smdebug.core.config_constants import (
    CLAIM_FILENAME,
    CONFIG_FILE_PATH_ENV_STR,
    DEFAULT_SAGEMAKER_OUTDIR,
    DEFAULT_SAGEMAKER_TENSORBOARD_PATH,
    TENSORBOARD_CONFIG_FILE_PATH_ENV_STR,
)
from smdebug.core.logger import get_logger
from smdebug.exceptions import IndexReaderException

logger = get_logger()


def ensure_dir(file_path, is_file=True):
    if is_file:
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_json_as_dict(s):
    if s is None or s == str(None):
        return None
    elif isinstance(s, str):
        return json.loads(s)
    elif isinstance(s, dict):
        return s
    else:
        raise ValueError("parameter must be either str or dict")


def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis


def split(comma_separated_string: str) -> List[str]:
    """Split "foo, bar,b az" into ["foo","bar","b az".]"""
    return [x.strip() for x in comma_separated_string.split(",")]


def merge_two_dicts(x, y) -> Dict:
    """If x and y have the same key, then y's value takes precedence.

    For example, merging
        x = {'a': 1, 'b': 2},
        y = {'b': 3, 'c': 4}
        yields
        z = {'a': 1, 'b': 3, 'c': 4}.
    """
    return {**x, **y}


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def is_s3(path):
    if path.startswith("s3://"):
        try:
            parts = path[5:].split("/", 1)
            return True, parts[0], parts[1]
        except IndexError:
            return True, path[5:], ""
    else:
        return False, None, None


def is_first_process(path):
    """
    This function is used to determine the caller of the process
    is the first process to do so.

    The purpose this function serves is to allow only one hook process
    to write in the case the user is unintentionally using the debugger
    with a training script that uses an unsupported distributed training
    process.

    In the case of s3 however, each hook process overwrites data written by the
    other hook processes and hence this is no possibility of race conditions,
    so this fn simply returns True.


    For non s3 mode, it uses the os.O_EXCL flag (https://linux.die.net/man/3/open)
    to determine if the the file already exists, i.e another process has
    written first.

    :param path: path to the trial
    :return: boolean that indicates if the caller was the
    first process to execute the fn.
    """
    s3, _, _ = is_s3(path)
    if s3:
        logger.debug(
            f"S3 Path passed to is_first_process. \
            {CLAIM_FILENAME} will not be generated."
        )
        return True  # Cannot Implement This Functionality for S3
    else:
        ensure_dir(path, is_file=False)
        filename = os.path.join(path, CLAIM_FILENAME)
        try:
            fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False


def remove_claim_file(path: str) -> None:
    """
    This function deletes the claim.smd file created by the is_first_process fn
    when the hook is closed.

    :param path: path to the trial
    :return: None
    """
    filename = os.path.join(path, CLAIM_FILENAME)
    s3, _, _ = is_s3(path)
    if s3:
        return
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def list_files_in_directory(directory, file_regex=None):
    files = []
    for root, dir_name, filename in os.walk(directory):
        for f in filename:
            if file_regex is None:
                files.append(os.path.join(root, f))
            elif re.match(file_regex, f):
                files.append(os.path.join(root, f))
    return files


def list_collection_files_in_directory(directory):
    import re

    collections_file_regex = re.compile(".*_?collections.json$")
    return list_files_in_directory(directory, file_regex=collections_file_regex)


def serialize_tf_device(device: str) -> str:
    """
    TF device strings have special characters that cannot be used in filenames.
    This function is used to convert those special characters/
    :param device:str
    :return: device:str
    """
    # _replica-0_task-0_device-GPU-0 = /replica:0/task:0/device:GPU:0
    device = device.replace("/", "_")
    device = device.replace(":", "-")
    return device


def deserialize_tf_device(device_name: str) -> str:
    """
    This function converts filenames back into tf device strings
    :param device_name: str
    :return: device_name: str
    """
    # /replica:0/task:0/device:GPU:0 = _replica-0_task-0_device-GPU-0
    device_name = device_name.replace("_", "/")
    device_name = device_name.replace("-", ":")
    return device_name


def match_inc(tname, include):
    """Matches anywhere in the string, doesn't require full match."""
    for inc in include:
        if re.search(inc, tname):
            return True
    return False


def index(sorted_list, elem):
    i = bisect.bisect_left(sorted_list, elem)
    if i != len(sorted_list) and sorted_list[i] == elem:
        return i
    raise ValueError


def get_region():
    # returns None if region is not set
    region_name = os.environ.get("AWS_REGION")
    if region_name is not None and region_name.strip() == "":
        region_name = None
    return region_name


def step_in_range(range_steps, step):
    if range_steps[0] is not None:
        begin = int(step) >= int(range_steps[0])
    else:
        begin = True
    if range_steps[1] is not None:
        end = int(step) < int(range_steps[1])
    else:
        end = True
    return begin and end


def get_relative_event_file_path(path):
    p = Path(path)
    path_parts = p.parts
    assert path_parts[-3] in ["events", "tensorboard"], str(path)
    return os.path.join(*path_parts[-3:])


def get_path_to_events_directory(path):
    return os.path.join(path, "events", "")


def size_and_shape(t):
    if type(t) == bytes or type(t) == str:
        return (len(t), [len(t)])
    return (t.nbytes, t.shape)


def get_path_to_collections(directory):
    collection_step = 0  # TODO: logic to determine collection steps
    return os.path.join(
        directory, "collections", format(collection_step, "09"), ""
    )  # The last "" adds a trailing /


def get_worker_name_from_collection_file(filename: str) -> str:
    """
    Extracts the worker name from the collection file.
    Collection files can currently have two formats:
        1. worker_0_collections.json
        2. _job-worker_replica-0_task-1_device-GPU-0_collections.json
    The leading underscore is used to indicate
    a distributed TF job worker in MirroredStrategy that needs to be deserialized.
    :param filename: str
    :return: worker_name: str
    """
    worker_name_regex = re.compile(".*/collections/.+/(.+)_collections.(json|ts)")
    worker_name = re.match(worker_name_regex, filename).group(1)
    if worker_name[0] == "_":
        worker_name = deserialize_tf_device(worker_name)
    return worker_name


def parse_worker_name_from_file(filename: str) -> str:
    """
    Extracts the worker name from the index or event file.
    Index / Event files can currently have two formats:
        1. (path_prefix)/(step_prefix)_worker_0.json
        2. (path_prefix)/(step_prefix)__replica-0_task-1_device-GPU-0.json
    The double underscore after step prefix is used to indicate
    a distributed TF job worker in MirroredStrategy that needs to be deserialized.
    :param filename: str
    :return: worker_name: str
    """
    # worker_2 = /tmp/ts-logs/index/000000001/000000001230_worker_2.json
    worker_name_regex = re.compile(r".+\/\d+_(.+)\.(json|csv|tfevents)$")
    worker_name_regex_match = re.match(worker_name_regex, filename)
    if worker_name_regex_match is None:
        raise IndexReaderException(f"Invalid File Found: {filename}")
    worker_name = worker_name_regex_match.group(1)
    if "__" in filename:
        # /replica:0/task:0/device:GPU:0 = replica-0_task-0_device-GPU-0.json
        worker_name = deserialize_tf_device(worker_name)
    return worker_name


def get_tb_worker():
    """Generates a unique string to be used as a worker name for tensorboard writers"""
    return f"{os.getpid()}_{socket.gethostname()}"


def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


class SagemakerSimulator(object):
    """
    Creates an environment variable pointing to a JSON config file, and creates the config file.
    Used for integration testing with zero-code-change.

    If `disable=True`, then we still create the `out_dir` directory, but ignore the config file.
    """

    def __init__(
        self,
        json_config_path="/tmp/zcc_config.json",
        tensorboard_dir="/tmp/tensorboard",
        training_job_name="sm_job",
        json_file_contents="{}",
        cleanup=True,
    ):
        self.out_dir = DEFAULT_SAGEMAKER_OUTDIR
        self.json_config_path = json_config_path
        self.tb_json_config_path = DEFAULT_SAGEMAKER_TENSORBOARD_PATH
        self.tensorboard_dir = tensorboard_dir
        self.training_job_name = training_job_name
        self.json_file_contents = json_file_contents
        self.cleanup = cleanup

    def __enter__(self):
        if self.cleanup is True:
            shutil.rmtree(self.out_dir, ignore_errors=True)
        shutil.rmtree(self.json_config_path, ignore_errors=True)
        tb_parent_dir = str(Path(self.tb_json_config_path).parent)
        shutil.rmtree(tb_parent_dir, ignore_errors=True)

        os.environ[CONFIG_FILE_PATH_ENV_STR] = self.json_config_path
        os.environ[TENSORBOARD_CONFIG_FILE_PATH_ENV_STR] = self.tb_json_config_path
        os.environ["TRAINING_JOB_NAME"] = self.training_job_name
        with open(self.json_config_path, "w+") as my_file:
            # We'll just use the defaults, but the file is expected to exist
            my_file.write(self.json_file_contents)

        os.makedirs(tb_parent_dir, exist_ok=True)
        with open(self.tb_json_config_path, "w+") as my_file:
            my_file.write(
                f"""
                {{
                    "LocalPath": "{self.tensorboard_dir}"
                }}
                """
            )

        return self

    def __exit__(self, *args):
        # Throws errors when the writers try to close.
        # shutil.rmtree(self.out_dir, ignore_errors=True)
        if self.cleanup is True:
            remove_file_if_exists(self.json_config_path)
            remove_file_if_exists(self.tb_json_config_path)
            if CONFIG_FILE_PATH_ENV_STR in os.environ:
                del os.environ[CONFIG_FILE_PATH_ENV_STR]
            if "TRAINING_JOB_NAME" in os.environ:
                del os.environ["TRAINING_JOB_NAME"]
            if TENSORBOARD_CONFIG_FILE_PATH_ENV_STR in os.environ:
                del os.environ[TENSORBOARD_CONFIG_FILE_PATH_ENV_STR]


class ScriptSimulator(object):
    def __init__(self, out_dir="/tmp/test", tensorboard_dir=None):
        self.out_dir = out_dir
        self.tensorboard_dir = tensorboard_dir

    def __enter__(self):
        shutil.rmtree(self.out_dir, ignore_errors=True)
        if self.tensorboard_dir:
            shutil.rmtree(self.tensorboard_dir, ignore_errors=True)
        return self

    def __exit__(self, *args):
        shutil.rmtree(self.out_dir, ignore_errors=True)
        if self.tensorboard_dir:
            shutil.rmtree(self.tensorboard_dir, ignore_errors=True)
