from .trial import EventFileTensor, Trial
from tornasole.core.utils import index
from tornasole.core.tfevent.util import EventFileLocation
from tornasole.core.collection_manager import CollectionManager
from tornasole.core.reader import FileReader
from tornasole_core.access_layer.utils import has_training_ended

import time
import os
import multiprocessing
import struct
from joblib import Parallel, delayed


class LocalTrial(Trial):
    def __init__(self, name, dirname,
                 range_steps=None, parallel=True,
                 check=False):
        super().__init__(name, range_steps=range_steps, parallel=parallel, check=check)
        dirname = os.path.expanduser(dirname)
        self.trial_dir = dirname
        self.logger.info(f'Loading trial {name} at path {self.trial_dir}')
        self.last_step_loaded = None
        self._load_collections()
        self._load_tensors()

    def _load_tensors(self):
        try:
            step_dirs = EventFileLocation.get_step_dirs(self.trial_dir)
        except FileNotFoundError:
            self.logger.debug('Waiting to see data for steps')
            return

        if self.range_steps is not None:
            step_dirs = [x for x in step_dirs if self._step_in_range(x)]

        step_dirs.sort()

        if self.last_step_loaded is not None:
            self.logger.debug("Trying to load events for steps after {}"
                              .format(int(self.last_step_loaded)))

            i = index(step_dirs, self.last_step_loaded)
            if i == len(step_dirs) - 1:
                # no new step
                return
            else:
                step_dirs = step_dirs[i+1:]

        self._read_step_dirs(step_dirs)

    def _load_collections(self):
        collections_file_path = os.path.join(self.trial_dir, 'collections.ts')
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

    def training_ended(self):
        return has_training_ended(self.trial_dir)

    def refresh_tensors(self):
        self._load_tensors()

    def __hash__(self):
        return hash((self.name, self.trial_dir))

    def __eq__(self, other):
        return (self.name, self.trial_dir) == (other.name, other.trial_dir)

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
            # Expected to be fixed with the introduction of indexreader
            try:
                dirnames_efts = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=0) \
                        (delayed(self._read_folder) \
                            (EventFileLocation.get_step_dir_path(self.trial_dir, step_dir),
                                read_data=self.read_data, check=self.check) \
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
                                                       read_data=self.read_data,
                                                       check=self.check))

        for dirname, dir_efts in dirnames_efts:
            self._add_tensors_at_steps(dir_efts)

        for dirname, efts in reversed(dirnames_efts):
            if len(efts) > 0:
                self.last_step_loaded = os.path.basename(dirname)
                break
            # make last_step_loaded equal to the newest dir which
            # had non zero tensors so that we can
            # look for newer steps with no tensors again.
            # note that if we load a non zero
            # number of tensors from a dir, we are guaranteed that
            # we can not load more tensors for that step since we use
            # temp file for writing event files and do atomic move

    @staticmethod
    def _read_folder(dirname, read_data=True, check=True):
        res = []
        for fname in os.listdir(dirname):
            if fname.endswith(".tfevents"):
                full_fname = os.path.join(dirname, fname)
                fr = FileReader(fname=full_fname)
                summary_values = fr.read_tensors(read_data=read_data, check=check)
                for sv in summary_values:
                    n, s, d, mode, mode_step = sv
                    eft = EventFileTensor(fname, tensor_name=n, step_num=s, tensor_value=d,
                                          mode=mode, mode_step=mode_step)
                    res.append(eft)

        return dirname, res
