import os
import re

from .utils import get_immediate_subdirectories, get_logger

logger = get_logger()


class TensorLocation:
    def __init__(self, tname, mode, mode_step, event_file_name, start_idx, length):
        self.tensorname = tname
        self.mode = mode
        self.mode_step = int(mode_step)
        self.event_file_name = event_file_name
        self.start_idx = start_idx
        self.length = length

    def to_dict(self):
        return {
            "tensorname": self.tensorname,
            "start_idx": self.start_idx,
            "length": self.length
        }


STEP_NUMBER_FORMATTING_LENGTH = '012'


class EventFileLocation:
    def __init__(self, step_num, worker_name, type='events'):
        self.step_num = int(step_num)
        self.worker_name = worker_name
        self.type = type

    def get_location(self, trial_dir=''):
        step_num_str = str(format(self.step_num, STEP_NUMBER_FORMATTING_LENGTH))
        event_filename = f"{step_num_str}_{self.worker_name}.tfevents"
        if trial_dir:
            event_key_prefix = os.path.join(trial_dir, self.type)
        else:
            event_key_prefix = self.type
        return os.path.join(event_key_prefix, step_num_str, event_filename)

    @staticmethod
    def match_regex(s):
        return EventFileLocation.load_filename(s, print_error=False)

    @staticmethod
    def load_filename(s, print_error=True):
        event_file_name = os.path.basename(s)
        m = re.search('(.*)_(.*).tfevents$', event_file_name)
        if m:
            step_num = int(m.group(1))
            worker_name = m.group(2)
            return EventFileLocation(step_num=step_num, worker_name=worker_name)
        else:
            if print_error:
                logger.error('Failed to load efl: ', s)
            return None

    @staticmethod
    def get_step_dirs(trial_dir):
        return get_immediate_subdirectories(os.path.join(trial_dir,
                                                         'events'))

    @staticmethod
    def get_step_dir_path(trial_dir, step_num):
        step_num = int(step_num)
        return os.path.join(trial_dir, 'events',
                            format(step_num, STEP_NUMBER_FORMATTING_LENGTH))


class IndexFileLocationUtils:
    # These functions are common to index reader and index writer
    MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX = 1000

    @staticmethod
    def get_index_prefix_for_step(step_num):
        index_prefix_for_step = step_num // IndexFileLocationUtils.MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
        return format(index_prefix_for_step, '09')

    @staticmethod
    def next_index_prefix_for_step(step_num):
        index_prefix_for_step = step_num // IndexFileLocationUtils.MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
        return format(index_prefix_for_step + 1, '09')

    @staticmethod
    def indexS3Key(trial_prefix, index_prefix_for_step_str, step_num, worker_name):
        step_num_str = format(step_num, '012')
        index_filename = format(f"{step_num_str}_{worker_name}.json")
        index_key = format(f"{trial_prefix}/index/{index_prefix_for_step_str}/{index_filename}")
        return index_key

    # for a step_num index files lies in prefix step_num/MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
    @staticmethod
    def get_index_key_for_step(trial_prefix, step_num, worker_name):
        index_prefix_for_step_str = IndexFileLocationUtils.get_index_prefix_for_step(step_num)
        return IndexFileLocationUtils.indexS3Key(trial_prefix, index_prefix_for_step_str, step_num, worker_name)
        # let's assume worker_name is given by hook
        # We need to think on naming conventions and access patterns for:
        # 1) muti-node training --> data parallel
        # 2) multi gpu training --> model parallel

    @staticmethod
    def get_step_from_idx_filename(index_file_name):
        i = index_file_name.find("_")
        k = index_file_name[i + 1:].find("_")
        step_num_str = index_file_name[i + 1: i + 1 + k]
        return int(step_num_str)

    @staticmethod
    def parse_step_from_index_file_name(index_file_name):
        # 10 = prefix/index/000000000/000000000010_worker.json'
        base_file_name = os.path.basename(index_file_name)
        step = int(base_file_name.split('_')[0])
        return step

    @staticmethod
    def get_index_path(path):
        return os.path.join(path, 'index')
