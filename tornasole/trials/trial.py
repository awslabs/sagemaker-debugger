import re
import time
from bisect import bisect_left
from abc import ABC, abstractmethod

from tornasole.core.access_layer.utils import has_training_ended
from tornasole.core.tensor import Tensor, StepState
from tornasole.exceptions import *
from tornasole.analysis.utils import refresh

from tornasole.core.locations import EventFileLocation
from tornasole.core.utils import flatten
from tornasole.core.logger import get_logger
from tornasole.core.reductions import TORNASOLE_REDUCTIONS_PREFIX, \
    reverse_reduction_tensor_name
from tornasole.core.modes import ModeKeys

from tornasole.core.locations import TensorLocation
from tornasole.core import index_reader


class EventFileTensor:
    def __init__(self, filename, tensor_name, step_num, tensor_value,
                 mode=None, mode_step=None):
        self.location = EventFileLocation.load_filename(filename)
        self.tensorname = tensor_name
        self.tensor_value = tensor_value
        self.step_num = step_num
        if mode is None:
            mode = ModeKeys.GLOBAL
        if mode_step is None:
            mode_step = step_num
        self.mode = mode
        self.mode_step = mode_step


class Trial(ABC):
    """
    Attributes:
        _tensors
        _index_tensors_dict

    ['name', '_tensors', '_mode_to_global', '_global_to_mode', 'logger', 'parallel',
    'check', 'range_steps', 'collection_manager', 'loaded_all_steps', 'cache', 'path',
    'index_tensors_dict', 'index_mode', 'last_event_token', 'last_index_token', 'index_reader',
    'dynamic_refresh', 'trial_dir']
    """

    def __init__(self, name, range_steps=None, parallel=True,
                 check=False, index_mode=True, cache=False):
        self.name = name
        self._tensors = {}

        # nested dictionary from mode -> mode_step -> global_step
        # will not have global mode as a key
        self._mode_to_global = {}

        # dictionary from global_step -> (mode, mode_step)
        # can have global mode as a value
        self._global_to_mode = {}

        self.logger = get_logger()
        self.parallel = parallel
        self.check = check
        self.range_steps = range_steps
        self.collection_manager = None
        self.loaded_all_steps = False
        self.cache = cache
        self.path = None
        self.index_tensors_dict = {}
        self.index_mode = index_mode
        self.last_event_token = None
        self.last_index_token = 0
        self.index_reader = index_reader.IndexReader

        # this is turned off during rule invocation for performance reasons since
        # required tensors are already fetched
        self.dynamic_refresh = True

        if self.range_steps is not None:
            assert self.range_steps[0] is None or \
                   (isinstance(self.range_steps[0], int) and self.range_steps[0] >= 0)
            assert self.range_steps[1] is None or \
                   (isinstance(self.range_steps[1], int) and self.range_steps[1] >= 0)
            if self.range_steps[1] is not None and self.range_steps[0] is not None:
                assert int(self.range_steps[1]) > int(self.range_steps[0]), "range_steps should be of the form " \
                    "(begin, end) where begin is less than end"
            if self.range_steps[0] is not None and self.range_steps[1] is not None:
                self.logger.info('Trial {} will look for steps between {} and {}'
                             .format(self.name, self.range_steps[0], self.range_steps[1]))

    @abstractmethod
    def _load_collections(self):
        pass

    @abstractmethod
    def _load_tensors_from_index_tensors(self, index_tensors_dict):
        pass

    @abstractmethod
    def _load_tensors_from_event_files(self, start_after_key=None):
        pass

    def __hash__(self):
        return hash((self.name, self.path))

    def __eq__(self, other):
        return (self.name, self.path) == (other.name, other.path)

    def maybe_refresh(self, name=None):
        if self.loaded_all_steps or not self.dynamic_refresh:
            return
        retry_count = 1
        training_ended = has_training_ended(self.path)
        if training_ended and self.loaded_all_steps== False:
            retry_count = 2
        while retry_count > 0:
            if name is None:
                self.refresh_tensors()
            else:
                self.refresh_tensor(name)
            if retry_count > 1:
                self.logger.info("Training has ended, will try to do a final refresh in 5 sec")
                time.sleep(5)
            retry_count -= 1
        if training_ended is True and self.loaded_all_steps is False:
            self.loaded_all_steps = True
            self.logger.info("Marked loaded all steps to True")

    def refresh_tensor(self, tname, steps=None):
        # for now we load all tensors at once
        self.refresh_tensors()

    def tensor(self, tname):
        # will not show tensor if it was not written yet
        # has tensor will refresh
        if self.has_tensor(tname):
            return self._tensors[tname]
        else:
            raise TensorUnavailable(tname)

    def has_tensor(self, tname):
        # will return false if tensor was not written yet
        if tname not in self._tensors:
            self.maybe_refresh(tname)
        return tname in self._tensors

    def _populate_step_dict(self, tensor_object, step_num):
        if tensor_object.mode != ModeKeys.GLOBAL:
            if tensor_object.mode not in self._mode_to_global:
                self._mode_to_global[tensor_object.mode] = {}
            if tensor_object.mode_step not in self._mode_to_global[tensor_object.mode]:
                self._mode_to_global[tensor_object.mode][tensor_object.mode_step] = int(step_num)
        if step_num not in self._global_to_mode:
            self._global_to_mode[step_num] = (tensor_object.mode, tensor_object.mode_step)

    def add_tensor(self, step_num, tensor_object):
        to = tensor_object
        # todo, use worker_name here
        if TORNASOLE_REDUCTIONS_PREFIX in to.tensorname:
            tname, red_name, abs = reverse_reduction_tensor_name(to.tensorname)
        else:
            tname = to.tensorname
        if tname not in self._tensors:
            t = Tensor(tname, trial=self, cache=self.cache)
            self._tensors[tname] = t
        t = self._tensors[tname]
        self._populate_step_dict(to, step_num)
        if TORNASOLE_REDUCTIONS_PREFIX in to.tensorname:
            if type(to) is TensorLocation:
                t.add_reduction_step_lazy(to.mode, to.mode_step,
                                 red_name, abs, to)
            else:
                t.add_reduction_step(to.mode, to.mode_step,
                                     red_name, abs, to.tensor_value)
        else:
            if type(to) is TensorLocation:
                t.add_step_lazy(to.mode, to.mode_step, to)
            else:
                t.add_step(to.mode, to.mode_step, to.tensor_value)

    def tensors(self):
        self.maybe_refresh()
        ts = list(self._tensors.keys())
        return ts

    def steps(self, mode=ModeKeys.GLOBAL):
        return self.available_steps(mode)

    def available_steps(self, mode=ModeKeys.GLOBAL):
        self.maybe_refresh()
        if mode == ModeKeys.GLOBAL:
            return sorted(self._global_to_mode.keys())
        elif mode in self._mode_to_global:
            return sorted(self._mode_to_global[mode].keys())
        else:
            return []

    def _global_step_currently(self, mode, mode_step):
        if mode == ModeKeys.GLOBAL:
            return mode_step
        elif mode in self._mode_to_global and \
          mode_step in self._mode_to_global[mode]:
            return self._mode_to_global[mode][mode_step]

    def global_step(self, mode, mode_step):
        s = self._global_step_currently(mode, mode_step)
        if s is not None:
            return s
        else:
            self.maybe_refresh()
            return self._global_step_currently(mode, mode_step)

    def _mode_modestep_currently(self, global_step):
        if global_step in self._global_to_mode:
            return self._global_to_mode[global_step]

    def mode_modestep(self, global_step):
        x = self._mode_modestep_currently(global_step)
        if x:
            return x
        else:
            self.maybe_refresh()
            x = self._mode_modestep_currently(global_step)
            if x:
                return x
        return None, None

    def mode_step(self, global_step):
        # can return global step itself in some cases
        x = self.mode_modestep(global_step)
        if x:
            return x[1]

    def mode(self, global_step):
        # can return global mode in some cases
        x = self.mode_modestep(global_step)
        if x:
            return x[0]

    def modes(self):
        # will not return global mode
        return self._mode_to_global.keys()

    def tensors_matching_regex(self, regex_list):
        self.maybe_refresh()
        matched_tensornames = []
        if not isinstance(regex_list, list):
            regex_list = [regex_list]
        regex_list = flatten(regex_list)
        for tensorname in self._tensors.keys():
            for regex_pattern in regex_list:
                if re.match(regex_pattern, tensorname):
                    matched_tensornames.append(tensorname)
                    break
        return matched_tensornames

    def collections(self):
        return self.collection_manager.collections

    def collection(self, coll_name):
        return self.collection_manager.get(coll_name)

    def tensors_in_collection(self, coll_name):
        rval = set()
        for x in self.collection(coll_name).get_tensor_names():
            rval.add(x)
        regex = self.collection(coll_name).get_include_regex()
        if regex:
            for x in self.tensors_matching_regex(regex):
                rval.add(x)
        return list(rval)

    def wait_for_steps(self, required_steps, mode=ModeKeys.GLOBAL):
        with refresh(self):
            for step in required_steps:
                while True:
                    s = self.has_passed_step(step, mode)
                    if s == StepState.UNAVAILABLE:
                        raise StepUnavailable(step, mode)
                    elif s == StepState.AVAILABLE:
                        break
                    elif self.loaded_all_steps is True:
                        last_step = -1
                        avail_steps = self.available_steps(mode=mode)
                        if len(avail_steps) > 0:
                            last_step = avail_steps[-1]
                        raise NoMoreData(step, mode, last_step)
                    time.sleep(5)

    def has_passed_step(self, step, mode=ModeKeys.GLOBAL):
        available_steps = self.available_steps(mode=mode)
        bisect_idx = bisect_left(available_steps, step)
        if bisect_idx < len(available_steps):
            if available_steps[bisect_idx] > step:
                return StepState.UNAVAILABLE
            elif available_steps[bisect_idx] == step:
                return StepState.AVAILABLE
        return StepState.NOT_YET_AVAILABLE

    def _add_tensors_at_steps(self, event_file_tensors):
        for eft in event_file_tensors:
            self.add_tensor(eft.step_num, tensor_object=eft)

    def load_tensors(self):
        if self.index_mode:
            self._load_tensors_from_index_files()
        else:
            self._load_tensors_from_event_files()

    def _load_tensors_from_index_files(self):
        self.index_tensors_dict, self.last_index_token = \
            self.index_reader.load_tensor_data_from_index_files(
                self.path,
                self.last_index_token,
                range_steps=self.range_steps)
        self._load_tensors_from_index_tensors(self.index_tensors_dict)

    def refresh_tensors(self):
        # TODO if job finished
        if self.index_mode:
            index_tensors_dict, self.last_index_token = \
                self.index_reader.load_tensor_data_from_index_files(self.path,
                                                                    start_after_key=self.last_index_token,
                                                                    range_steps=self.range_steps)
            if len(index_tensors_dict):
                self.index_tensors_dict.update(index_tensors_dict)
                self._load_tensors_from_index_tensors(index_tensors_dict)
        else:
            self._load_tensors_from_event_files(start_after_key=self.last_event_token)
