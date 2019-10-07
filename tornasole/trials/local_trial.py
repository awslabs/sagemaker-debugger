from .trial import EventFileTensor, Trial

from tornasole.core.utils import index, step_in_range
from tornasole.core.locations import EventFileLocation
from tornasole.core.collection_manager import CollectionManager, \
    COLLECTIONS_FILE_NAME
from tornasole.core.reader import FileReader

import time
import os
import multiprocessing
import struct
from joblib import Parallel, delayed


class LocalTrial(Trial):
    def __init__(self, name, dirname,
                 range_steps=None, parallel=True,
                 check=False,
                 index_mode=True,
                 cache=False):
        super().__init__(name, range_steps=range_steps, parallel=parallel,
                         check=check, index_mode=index_mode, cache=cache)
        self.path = os.path.expanduser(dirname)
        self.trial_dir = self.path
        self.logger.info(f'Loading trial {name} at path {self.trial_dir}')
        self._load_collections()
        self.load_tensors()

    def _load_tensors_from_index_tensors(self, index_tensors_dict):
        for tname in index_tensors_dict:
            for step, itds in index_tensors_dict[tname].items():
                self.add_tensor(int(step), itds['tensor_location'])

    def _load_tensors_from_event_files(self):
        try:
            step_dirs = EventFileLocation.get_step_dirs(self.trial_dir)
        except FileNotFoundError:
            self.logger.debug('Waiting to see data for steps')
            return

        if self.range_steps is not None:
            step_dirs = [x for x in step_dirs if step_in_range(self.range_steps, x)]

        step_dirs.sort()

        if self.last_event_token:
            self.logger.debug("Trying to load events for steps after {}"
                              .format(int(self.last_event_token)))
            i = index(step_dirs, self.last_event_token)

            if i == len(step_dirs) - 1:
                # no new step
                return
            else:
                step_dirs = step_dirs[i + 1:]

        self._read_step_dirs(step_dirs)

    def _load_collections(self):
        collections_file_path = os.path.join(self.trial_dir, COLLECTIONS_FILE_NAME)
        num_times_before_warning = 10
        while True:
            if os.path.exists(collections_file_path):
                self.collection_manager = CollectionManager.load(collections_file_path)
                self.logger.info(f'Loaded {len(self.collection_manager.collections)} collections')
                break
            else:
                time.sleep(2)
                num_times_before_warning -= 1
                if num_times_before_warning < 0:
                    self.logger.warning('Waiting to read collections')
                else:
                    self.logger.debug('Waiting to read collections')
                continue

    def get_tensors(self, tname_steps_dict, should_regex_match=False):
        # now we do not need to do anything since we read the full event file
        pass

    def _read_step_dirs(self, step_dirs):
        if len(step_dirs) == 0:
            return

        dirnames_efts = []
        if self.parallel:
            # Ugly hack for https://github.com/awslabs/tornasole_rules/issues/66
            # Temp fix with intentional code duplication
            # Expected to be fixed with the introduction of index_reader
            try:
                dirnames_efts = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=0) \
                    (delayed(self._read_folder) \
                         (EventFileLocation.get_step_dir_path(self.trial_dir, step_dir),
                          check=self.check) \
                     for step_dir in step_dirs)
                # sort them as parallel returns in random order
                # we want to sort them by dirname
                dirnames_efts.sort(key=lambda x: int(os.path.basename(x[0])))
            except struct.error:
                self.logger.warning('Failed to load with parallel. Loading events serially now')
                self.parallel = False
                self._read_step_dirs(step_dirs)
        else:
            for step_dir in step_dirs:
                step_dir_path = EventFileLocation.get_step_dir_path(self.trial_dir, step_dir)
                dirnames_efts.extend(self._read_folder(step_dir_path,
                                                       check=self.check))

        for dirname, dir_efts in dirnames_efts:
            self._add_tensors_at_steps(dir_efts)

        for dirname, efts in reversed(dirnames_efts):
            if len(efts) > 0:
                self.last_event_token = os.path.basename(dirname)
                break
            # make last_event_token equal to the newest dir which
            # had non zero tensors so that we can
            # look for newer steps with no tensors again.
            # note that if we load a non zero
            # number of tensors from a dir, we are guaranteed that
            # we can not load more tensors for that step since we use
            # temp file for writing event files and do atomic move

    @staticmethod
    def _read_folder(dirname, check=True):
        res = []
        for fname in os.listdir(dirname):
            if fname.endswith(".tfevents"):
                full_fname = os.path.join(dirname, fname)
                fr = FileReader(fname=full_fname)
                summary_values = fr.read_tensors(check=check)
                for sv in summary_values:
                    n, s, d, mode, mode_step = sv
                    eft = EventFileTensor(fname, tensor_name=n, step_num=s, tensor_value=d,
                                          mode=mode, mode_step=mode_step)
                    res.append(eft)
        return dirname, res
