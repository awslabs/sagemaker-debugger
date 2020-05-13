# Standard Library
import os
import re
from abc import ABC, abstractmethod

# Local
from .logger import get_logger
from .utils import get_immediate_subdirectories

logger = get_logger()


class TensorLocation:
    def __init__(self, tname, mode, mode_step, event_file_name, start_idx, length, worker):
        self.tensorname = tname
        self.mode = mode
        self.mode_step = int(mode_step)
        self.event_file_name = event_file_name
        self.start_idx = start_idx
        self.length = length
        self.worker = worker

    def to_dict(self):
        return {"tensorname": self.tensorname, "start_idx": self.start_idx, "length": self.length}


STEP_NUMBER_FORMATTING_LENGTH = "012"


class EventFileLocation(ABC):
    def __init__(self, step_num, worker_name):
        self.step_num = int(step_num)
        self.worker_name = worker_name
        self.type = None

    def get_step_num_str(self):
        return str(format(self.step_num, STEP_NUMBER_FORMATTING_LENGTH))

    def get_filename(self):
        step_num_str = self.get_step_num_str()
        event_filename = f"{step_num_str}_{self.worker_name}.tfevents"
        return event_filename

    @classmethod
    def match_regex(cls, s):
        return cls.load_filename(s, print_error=False)

    @classmethod
    def load_filename(cls, s, print_error=True):
        event_file_name = os.path.basename(s)
        m = re.search("(.*)_(.*).tfevents$", event_file_name)
        if m:
            step_num = int(m.group(1))
            worker_name = m.group(2)
            return cls(step_num=step_num, worker_name=worker_name)
        else:
            if print_error:
                logger.error("Failed to load efl: ", s)
            return None

    @abstractmethod
    def get_file_location(self):
        pass


class TensorFileLocation(EventFileLocation):
    def __init__(self, step_num, worker_name):
        super().__init__(step_num, worker_name)
        self.type = "events"

    @staticmethod
    def get_dir(trial_dir):
        return os.path.join(trial_dir, "events")

    def get_file_location(self, trial_dir=""):
        if trial_dir:
            event_key_prefix = self.get_dir(trial_dir)
        else:
            event_key_prefix = self.type
        return os.path.join(event_key_prefix, self.get_step_num_str(), self.get_filename())

    @classmethod
    def get_step_dirs(cls, trial_dir):
        return get_immediate_subdirectories(cls.get_dir(trial_dir))

    @classmethod
    def get_step_dir_path(cls, trial_dir, step_num):
        step_num = int(step_num)
        return os.path.join(cls.get_dir(trial_dir), format(step_num, STEP_NUMBER_FORMATTING_LENGTH))


class TensorboardFileLocation(EventFileLocation):
    def __init__(self, step_num, worker_name, mode=None):
        super().__init__(step_num, worker_name)
        self.mode = mode
        self.type = "tensorboard"

    def get_file_location(self, base_dir=""):
        # when base_dir is empty it just returns the relative file path
        if base_dir:
            event_key_prefix = os.path.join(base_dir, self.mode.name)
        else:
            event_key_prefix = os.path.join(self.type, self.mode.name)

        return os.path.join(event_key_prefix, self.get_filename())


class IndexFileLocationUtils:
    # These functions are common to index reader and index writer
    MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX = 1000

    @staticmethod
    def get_index_prefix_for_step(step_num):
        index_prefix_for_step = (
            step_num // IndexFileLocationUtils.MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
        )
        return format(index_prefix_for_step, "09")

    @staticmethod
    def next_index_prefix_for_step(step_num):
        index_prefix_for_step = (
            step_num // IndexFileLocationUtils.MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
        )
        return format(index_prefix_for_step + 1, "09")

    @staticmethod
    def _get_index_key(trial_prefix, step_num, worker_name):
        index_prefix_for_step_str = IndexFileLocationUtils.get_index_prefix_for_step(step_num)
        step_num_str = format(step_num, "012")
        index_filename = format(f"{step_num_str}_{worker_name}.json")
        index_key = os.path.join(trial_prefix, "index", index_prefix_for_step_str, index_filename)
        return index_key

    # for a step_num index files lies
    # in prefix step_num/MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
    @staticmethod
    def get_index_key_for_step(trial_prefix, step_num, worker_name):
        return IndexFileLocationUtils._get_index_key(trial_prefix, step_num, worker_name)

    @staticmethod
    def get_step_from_idx_filename(index_file_name):
        i = index_file_name.find("_")
        k = index_file_name[i + 1 :].find("_")
        step_num_str = index_file_name[i + 1 : i + 1 + k]
        return int(step_num_str)

    @staticmethod
    def parse_step_from_index_file_name(index_file_name):
        # 10 = prefix/index/000000000/000000000010_worker.json'
        # 10 = prefix/index/000000000/000000000010_worker_EVAL.json'
        base_file_name = os.path.basename(index_file_name)
        step = int(base_file_name.split("_")[0])
        return step

    @staticmethod
    def get_index_path(path):
        return os.path.join(path, "index")

    @staticmethod
    def get_prefix_from_index_file(index_file: str) -> str:
        """
        The function returns the filepath prefix before 'index/' in the
        the index_file names.

        For example:
            get_prefix_from_index_file("prefix/index/000000000/000000000010_worker.json")

            will return prefix
        :param index_file: str
        :return: str
        """
        # prefix = prefix/index/000000000/000000000010_worker.json'
        r = re.compile("(.+)/index/.+$")
        return re.match(r, index_file).group(1)
