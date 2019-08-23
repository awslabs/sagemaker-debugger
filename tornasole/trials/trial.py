import re
import time
from bisect import bisect_left
from abc import ABC, abstractmethod

from tornasole.core.tensor import Tensor, StepState
from tornasole.exceptions import *
from tornasole.analysis.utils import refresh

from tornasole.core.tfevent.util import EventFileLocation
from tornasole.core.utils import get_logger, flatten
from tornasole.core.reductions import TORNASOLE_REDUCTIONS_PREFIX, reverse_reduction_tensor_name
from tornasole.core.modes import ModeKeys


class EventFileTensor:
    def __init__(self, filename, tensor_name, step_num, tensor_value,
                 mode=None, mode_step=None):
        self.location = EventFileLocation.load_filename(filename)
        self.tensor_name = tensor_name
        self.tensor_value = tensor_value
        self.step_num = step_num
        if mode is None:
            mode = ModeKeys.GLOBAL
        if mode_step is None:
            mode_step = step_num
        self.mode = mode
        self.mode_step = mode_step


class Trial(ABC):
    def __init__(self, name, range_steps=None, parallel=True, read_data=True, check=False):
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
        self.read_data = read_data
        self.check = check
        self.range_steps = range_steps
        self.collection_manager = None
        self.loaded_all_steps = False

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
    def _load_tensors(self):
        pass

    @abstractmethod
    def _load_collections(self):
        pass

    @abstractmethod
    def refresh_tensors(self):
        pass

    @abstractmethod
    def training_ended(self):
        pass

    def maybe_refresh(self, name=None):
        
        if self.loaded_all_steps or not self.dynamic_refresh:
            return
        retry_count = 1
        training_ended = self.training_ended()
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
        if training_ended == True and self.loaded_all_steps == False:
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

    def set_tensor_value(self, event_file_tensor):
        eft = event_file_tensor
        # todo, use worker_name here

        if TORNASOLE_REDUCTIONS_PREFIX in eft.tensor_name:
            tname, red_name, abs = reverse_reduction_tensor_name(eft.tensor_name)
        else:
            tname = eft.tensor_name

        if tname not in self._tensors:
            t = Tensor(tname, trial=self)
            self._tensors[tname] = t
        t = self._tensors[tname]

        if eft.mode != ModeKeys.GLOBAL:
            if eft.mode not in self._mode_to_global:
                self._mode_to_global[eft.mode] = {}
            if eft.mode_step not in self._mode_to_global[eft.mode]:
                self._mode_to_global[eft.mode][eft.mode_step] = eft.step_num

        if eft.step_num not in self._global_to_mode:
            self._global_to_mode[eft.step_num] = (eft.mode, eft.mode_step)

        if 'tornasole/reductions/' in eft.tensor_name:
            t.add_reduction_step(eft.mode, eft.mode_step,
                                 red_name, abs, eft.tensor_value)
        else:
            t.add_step(eft.mode, eft.mode_step, eft.tensor_value)

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
        for x in self.collection(coll_name).get_reduction_tensor_names():
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
                    elif self.loaded_all_steps == True:
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
            self.set_tensor_value(eft)

    def _step_in_range(self, x):
        if self.range_steps[0] is not None:
            begin = int(x) >= int(self.range_steps[0])
        else:
            begin = True
        if self.range_steps[1] is not None:
            end = int(x) < int(self.range_steps[1])
        else:
            end = True
        return begin and end