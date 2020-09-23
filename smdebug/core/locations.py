# Standard Library
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime

# First Party
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    DEFAULT_PREFIX,
    MERGEDTIMELINE_SUFFIX,
    PYTHONTIMELINE_SUFFIX,
    TRACE_DIRECTORY_FORMAT,
)

# Local
from .logger import get_logger
from .utils import ensure_dir, get_immediate_subdirectories, get_node_id

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


class TensorShape:
    def __init__(self, name, mode, mode_step, shape, original_name=None):
        self.name = name
        self.original_name = original_name if original_name is not None else name
        self.mode = mode
        self.mode_step = mode_step
        self.shape = tuple(shape)

    def to_dict(self):
        return {"tensorname": self.name, "originalname": self.original_name, "shape": self.shape}


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


class TraceFileLocation:
    # File path generated based on
    # $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/
    # $FILEEVENTENDTIMEUTCINEPOCH_{$ENV_NODE_ID}_model_timeline.json
    @staticmethod
    def get_file_location(timestamp, base_dir, suffix=PYTHONTIMELINE_SUFFIX):
        env_base_location = base_dir
        date_hour = time.strftime(
            TRACE_DIRECTORY_FORMAT, time.gmtime(timestamp / CONVERT_TO_MICROSECS)
        )
        timestamp = int(round(timestamp))
        worker_id = get_node_id()
        file_path = os.path.join(
            env_base_location,
            DEFAULT_PREFIX
            + "/"
            + date_hour
            + "/"
            + str(timestamp)
            + "_"
            + worker_id
            + "_"
            + suffix,
        )
        return file_path

    @staticmethod
    def get_detailed_profiling_log_dir(base_folder, framework, current_step):
        current_time = datetime.today().strftime("%Y%m%d%H")
        padded_start_step = str(current_step).zfill(9)
        log_dir = os.path.join(
            base_folder,
            "framework",
            framework,
            "detailed_profiling",
            current_time,
            padded_start_step,
        )
        ensure_dir(log_dir, is_file=False)
        return log_dir

    @staticmethod
    def get_tf_profiling_metadata_file(base_folder, start_time_us, end_time_us):
        metadata_file = os.path.join(
            base_folder,
            get_node_id()
            + "_"
            + str(int(round(start_time_us)))
            + "_"
            + str(int(round(end_time_us)))
            + ".metadata",
        )
        ensure_dir(metadata_file, is_file=True)
        return metadata_file

    @staticmethod
    def get_python_profiling_stats_dir(
        base_folder,
        profiler_name,
        framework,
        start_mode,
        start_step,
        start_phase,
        start_time_since_epoch_in_micros,
        end_mode,
        end_step,
        end_phase,
        end_time_since_epoch_in_micros,
    ):
        node_id = get_node_id()
        stats_dir = "{0}-{1}-{2}-{3}_{4}-{5}-{6}-{7}".format(
            start_mode,
            start_step,
            start_phase,
            start_time_since_epoch_in_micros,
            end_mode,
            end_step,
            end_phase,
            end_time_since_epoch_in_micros,
        )
        stats_dir_path = os.path.join(
            base_folder, "framework", framework, profiler_name, node_id, stats_dir
        )
        ensure_dir(stats_dir_path, is_file=False)
        return stats_dir_path

    @staticmethod
    def get_merged_trace_file_location(base_dir, timestamp_in_us):
        env_base_location = base_dir
        timestamp = int(round(timestamp_in_us))
        worker_id = get_node_id()
        file_path = os.path.join(
            env_base_location, str(timestamp) + "_" + worker_id + "_" + MERGEDTIMELINE_SUFFIX
        )
        return file_path


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
