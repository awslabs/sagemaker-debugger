from tornasole.core.reductions import get_numpy_reduction
from tornasole.core.modes import ModeKeys
import bisect
from tornasole.exceptions import *

from enum import Enum


class StepState(Enum):
    UNAVAILABLE = 0
    AVAILABLE = 1
    NOT_YET_AVAILABLE = 2


class ModeSteps:
    def __init__(self, mode):
        self.mode = mode
        self._steps = {}

    def steps(self):
        ts = list(self._steps.keys())
        ts.sort(key=int)
        return ts

    def has_step(self, step_num):
        return step_num in self._steps

    def set_step_value(self, step_num, value):
        if step_num not in self._steps:
            self._steps[step_num] = Step(step_num, value)
        else:
            s = self._steps[step_num]
            s.value = value

    def set_step_reduction_value(self, step_num, red_name, abs, red_value):
        if step_num not in self._steps:
            s = Step(step_num)
            self._steps[step_num] = s
        else:
            s = self._steps[step_num]
        s.set_reduction_value(red_name, abs, red_value)

    def step(self, step_num):
        return self._steps[step_num]


class Step:
    def __init__(self, step_num, value=None):
        self.step_num = step_num
        self._value = value

        # mapping from (red_name, abs) to value
        self._reduction_values = {}

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @value.deleter
    def value(self):
        del self._value

    def reduction_values(self):
        return self._reduction_values

    def reduction_value(self, red_name, abs):
        if (red_name, abs) in self._reduction_values:
            return self._reduction_values[(red_name, abs)]

    def set_reduction_value(self, red_name, abs, red_value):
        self._reduction_values[(red_name, abs)] = red_value


# refreshing is always responsibility of tensor class at the highest level API function,
# not ModeSteps/Steps
class Tensor:
    def __init__(self, name, trial):
        self._mode_steps = {}
        self.name = name
        self.trial = trial

    def steps(self, mode=ModeKeys.GLOBAL):
        self.trial.maybe_refresh(self.name)
        if mode == ModeKeys.GLOBAL:
            return self._global_steps()
        elif mode in self._mode_steps:
            return self._mode_steps[mode].steps()
        else:
            return None

    def _global_steps(self):
        gs = []
        for mode in self._mode_steps:
            ms = self._mode_steps[mode].steps()
            for s in ms:
                gs.append(self.trial.global_step(mode, s))
        gs.sort(key=int)
        return gs

    def _has_step(self, step_num, mode=ModeKeys.GLOBAL):
        if self._has_step_currently(step_num, mode):
            return True
        else:
            self.trial.maybe_refresh(self.name)
            if self._has_step_currently(step_num, mode):
                return True
        return False

    def _has_step_currently(self, step_num, mode):
        if mode == ModeKeys.GLOBAL:
            return self._has_global_step_currently(step_num)
        else:
            return self._has_mode_step_currently(step_num, mode)

    def _has_mode_step_currently(self, step_num, mode):
        if mode in self._mode_steps:
            if self._mode_steps[mode].has_step(step_num):
                return True
        return False

    def _has_global_step_currently(self, step_num):
        # first check if in global mode,
        if ModeKeys.GLOBAL in self._mode_steps:
            if self._mode_steps[ModeKeys.GLOBAL].has_step(step_num):
                return True
        else:
            # else convert to mode_step and check
            mode, mode_step_num = self.trial.mode_modestep(step_num)
            if mode in self._mode_steps and \
              self._mode_steps[mode].has_step(mode_step_num):
                return True
        return False

    def _get_step_currently(self, step_num, mode):
        if mode == ModeKeys.GLOBAL and ModeKeys.GLOBAL in self._mode_steps \
          and self._mode_steps[ModeKeys.GLOBAL].has_step(step_num):
            # step was saved as GLOBAL step
            return self._mode_steps[mode].step(step_num)
        else:
            if mode == ModeKeys.GLOBAL:
                # else convert to mode_step and check
                mode, step_num = self.trial.mode_modestep(step_num)
            if self._has_mode_step_currently(step_num, mode):
                return self._mode_steps[mode].step(step_num)
        return None

    def step(self, step_num, mode=ModeKeys.GLOBAL):
        s = self._get_step_currently(step_num, mode)
        if s is not None:
            return s
        else:
            self.trial.maybe_refresh(self.name)
            ss = self.trial.has_passed_step(step_num, mode)
            if ss == StepState.AVAILABLE:
                s = self._get_step_currently(step_num, mode)
                if s is not None:
                    return s
                raise TensorUnavailableForStep(self.name, step_num, mode)
            elif ss == StepState.UNAVAILABLE:
                raise StepUnavailable(step_num, mode)
            elif ss == StepState.NOT_YET_AVAILABLE:
                if self.trial.loaded_all_steps == True:
                    last_step = -1
                    avail_steps = self.trial.available_steps(mode=mode)
                    if len(avail_steps) > 0:
                        last_step = avail_steps[-1]
                    raise NoMoreData("Looking for step:{} for mode {} and reached end of training. Max step available is {}".format(step_num, mode, last_step))
                raise StepNotYetAvailable(step_num, mode)
        assert False, 'Should not happen'

    def value(self, step_num, mode=ModeKeys.GLOBAL):
        # step refreshes
        s = self.step(step_num=step_num, mode=mode)
        if s.value is not None:
            return s.value
        else:
            has_reductions = len(s.reduction_values()) > 0
            raise TensorUnavailableForStep(self.name, step_num, mode, has_reductions)

    def reduction_values(self, step_num, mode=ModeKeys.GLOBAL):
        s = self.step(step_num=step_num, mode=mode)
        if s is not None:
            return s.reduction_values()
        else:
            assert False, 'Should not happen'

    def reduction_value(self, step_num, reduction_name, mode=ModeKeys.GLOBAL, abs=False):
        """
        Returns the value of the reduction requested.
        If the tensor was saved as a reduction, then just fetches that.
        Else, tries to compute the reduction and returns. If the tensor value is not
        available, returns None as reduction

        :param step_num: step number
        :param mode: mode of job (train, eval, predict, etc).
                            If this is None, assumes step number is global
        :param reduction_name: name of reduction
        :param abs: boolean which represents whether reduction should
                    be applied on absolute value of the tensor or not
        :return: reduction value requested as a float
        """
        s = self.step(step_num=step_num, mode=mode)
        rv = s.reduction_value(reduction_name, abs)
        if rv is not None:
            return rv
        elif s.value is not None:
            return get_numpy_reduction(reduction_name, s.value, abs)

        assert False, 'Should not happen'

    def _create_mode_step(self, mode, mode_step):
        mode_step = int(mode_step)
        if mode_step < 0:
            raise ValueError('mode step number {} for tensor {} '
                             'can not be less than 0'.format(mode_step, self.name))
        if mode not in self._mode_steps:
            self._mode_steps[mode] = ModeSteps(mode)

    def add_step(self, mode, mode_step, value):
        self._create_mode_step(mode, mode_step)
        self._mode_steps[mode].set_step_value(mode_step, value)

    def add_reduction_step(self, mode, mode_step, red_name, abs, red_value):
        self._create_mode_step(mode, mode_step)
        self._mode_steps[mode].set_step_reduction_value(mode_step,
                                                        red_name, abs, red_value)

    def prev_steps(self, step, n, mode=ModeKeys.GLOBAL):
        """
        returns n prev steps from step representing step number
        of given mode
        :param step: int
        step number
        :param n: int
        number of previous steps to return
        :param mode: value of the enum tornasole.modes
        modes.GLOBAL, modes.TRAIN, modes.EVAL, modes.PREDICT
        :return: a list of step numbers
        """
        steps = self.steps(mode=mode)
        i = bisect.bisect_right(steps, step)
        prev_steps = steps[:i]
        return prev_steps[-n:]
