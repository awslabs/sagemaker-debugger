# Standard Library
import bisect
from enum import Enum
from typing import Dict, Tuple

# Third Party
import numpy as np

# First Party
from smdebug.exceptions import (
    InvalidWorker,
    NoMoreData,
    ShapeUnavailableForStep,
    StepNotYetAvailable,
    StepUnavailable,
    TensorUnavailableForStep,
)

# Local
from .locations import TensorLocation
from .modes import ModeKeys
from .reductions import get_numpy_reduction


class StepState(Enum):
    UNAVAILABLE = 0
    AVAILABLE = 1
    NOT_YET_AVAILABLE = 2


class ModeSteps:
    """Contains a ModeKey and a dictionary mapping step numbers to a dictionary of workers to steps."""

    def __init__(self, mode):
        self.mode = mode
        self._steps = {}

    def steps(self):
        step_nums = list(self._steps.keys())
        step_nums.sort(key=int)
        return step_nums

    def has_step(self, step_num):
        return step_num in self._steps

    def set_step_value(self, step_num, worker, value):
        step = Step(step_num, value=value)
        if step_num not in self._steps:
            self._steps[step_num] = {worker: step}
        elif worker not in self._steps[step_num]:
            self._steps[step_num].update({worker: step})

        s = self._steps[step_num][worker]
        s.value = value

    def set_step_location(self, step_num, worker, location):
        step = Step(step_num, location=location)
        if step_num not in self._steps:
            self._steps[step_num] = {worker: step}
        elif worker not in self._steps[step_num]:
            self._steps[step_num].update({worker: step})

        s = self._steps[step_num][worker]
        s.location = location

    def set_step_shape(self, step_num, worker, shape):
        step = Step(step_num, shape=shape)
        if step_num not in self._steps:
            self._steps[step_num] = {worker: step}
        elif worker not in self._steps[step_num]:
            self._steps[step_num].update({worker: step})

        s = self._steps[step_num][worker]
        s.shape = shape

    def set_step_reduction_value(self, step_num, worker, red_name, abs, red_value):
        if step_num not in self._steps:
            s = Step(step_num)
            self._steps[step_num] = {worker: s}
        elif worker not in self._steps[step_num]:
            s = Step(step_num)
            self._steps[step_num].update({worker: s})
        s = self._steps[step_num][worker]
        s.set_reduction_value(red_name, abs, red_value)

    def set_step_reduction_location(self, step_num, worker, red_name, abs, red_location):
        if step_num not in self._steps:
            self._steps[step_num] = {worker: Step(step_num)}
        elif worker not in self._steps[step_num]:
            s = Step(step_num)
            self._steps[step_num].update({worker: s})
        s = self._steps[step_num][worker]
        s.set_reduction_location(red_name, abs, red_location)

    def step(self, step_num):
        return self._steps[step_num]


class Step:
    """Contains the step number, value, location, and reduction values/locations."""

    def __init__(self, step_num, value=None, location=None, shape=None):
        self.step_num = step_num
        self.value = value
        self.location = location
        self.shape = shape

        # mapping from (red_name, abs) to value
        self._reduction_values = {}
        self._reduction_locations = {}

    def reduction_values(self) -> Dict[Tuple[str, bool], np.ndarray]:
        """Return a dictionary mapping reduction tuples to floats."""
        return self._reduction_values

    def reduction_value(self, red_name: str, abs: bool) -> np.ndarray:
        """Return the value for a single reduction as a NumPy array."""
        if (red_name, abs) in self._reduction_values:
            return self._reduction_values[(red_name, abs)]

    def reduction_locations(self) -> Dict[Tuple[str, bool], TensorLocation]:
        return self._reduction_locations

    def reduction_location(self, red_name: str, abs: bool) -> TensorLocation:
        if (red_name, abs) in self._reduction_locations:
            return self._reduction_locations[(red_name, abs)]

    def set_reduction_value(self, red_name: str, abs: bool, red_value: np.ndarray):
        self._reduction_values[(red_name, abs)] = red_value

    def set_reduction_location(self, red_name: str, abs: bool, red_location: TensorLocation):
        self._reduction_locations[(red_name, abs)] = red_location


# refreshing is always responsibility of tensor class at the highest level API function,
# not ModeSteps/Steps
class Tensor:
    def __init__(self, name, trial, cache):
        self._mode_steps = {}
        self.name = name
        # In TF Keras and Variables in all interfaces of TF,
        # SMDebug modifies some names of tensors to be more descriptive.
        # In such cases we save here the original name.
        self.original_name = None
        self.trial = trial
        self.cache = cache

    def steps(self, mode=ModeKeys.GLOBAL, show_incomplete_steps=False) -> list:
        """
        the steps function call returns only completed steps to
        the user.
        :param mode: ModeKeys
        :param show_incomplete_steps: bool
        :return: list
        """
        all_steps = self._all_steps(mode)
        if show_incomplete_steps is True:
            return all_steps
        completed_steps = list()
        for step in all_steps:
            if (
                self.workers(step, mode) == self.trial.num_workers
                or self.trial.loaded_all_steps is True
                or self.trial.last_complete_step >= step
            ):
                completed_steps.append(step)
        return completed_steps

    def _all_steps(self, mode=ModeKeys.GLOBAL) -> list:
        """
        the all_steps function call returns all the steps,
        complete or incomplete the user.
        :param mode: ModeKeys
        :return: list
        """
        self.trial.maybe_refresh()
        if mode == ModeKeys.GLOBAL:
            return self._global_steps()
        elif mode in self._mode_steps:
            return self._mode_steps[mode].steps()
        else:
            return []

    def _global_steps(self):
        gs = []
        for mode in self._mode_steps:
            ms = self._mode_steps[mode].steps()
            for s in ms:
                gs.append(self.trial.global_step(mode, s))
        gs.sort(key=int)
        return gs

    def _has_mode_step_currently(self, step_num, mode):
        if mode in self._mode_steps:
            if self._mode_steps[mode].has_step(step_num):
                return True
        return False

    def _get_step_dict(self, step_num, mode):
        if (
            mode == ModeKeys.GLOBAL
            and ModeKeys.GLOBAL in self._mode_steps
            and self._mode_steps[ModeKeys.GLOBAL].has_step(step_num)
        ):
            # step was saved as GLOBAL step
            return self._mode_steps[mode].step(step_num)
        else:
            if mode == ModeKeys.GLOBAL:
                # else convert to mode_step and check
                mode, step_num = self.trial.mode_modestep(step_num)
            if self._has_mode_step_currently(step_num, mode):
                return self._mode_steps[mode].step(step_num)
        return None

    def _get_step_currently(self, step_num, mode, worker=None) -> Step:
        step_dict = self._get_step_dict(step_num, mode)
        if step_dict is not None:
            if worker and worker not in step_dict:
                raise InvalidWorker(worker)
            if worker is None:
                workers = sorted(step_dict.keys())
                assert len(workers) > 0
                worker = workers[0]
            return step_dict[worker]
        return None

    def step(self, step_num, mode=ModeKeys.GLOBAL, worker=None):
        raise NotImplementedError(
            "step method has been removed. Please use tensor.value "
            "or tensor.reduction_value methods"
        )

    def _step(self, step_num, mode=ModeKeys.GLOBAL, worker=None):
        s = self._get_step_currently(step_num, mode, worker=worker)
        if s is not None:
            return s
        else:
            self.trial.maybe_refresh(self.name)
            ss = self.trial.has_passed_step(step_num, mode)
            if ss == StepState.AVAILABLE:
                s = self._get_step_currently(step_num, mode, worker=worker)
                if s is not None:
                    return s
                raise TensorUnavailableForStep(self.name, step_num, mode)
            elif ss == StepState.UNAVAILABLE:
                raise StepUnavailable(step_num, mode)
            elif ss == StepState.NOT_YET_AVAILABLE:
                if self.trial.loaded_all_steps is True:
                    last_step = -1
                    avail_steps = self.trial.steps(mode=mode)
                    if len(avail_steps) > 0:
                        last_step = avail_steps[-1]
                    raise NoMoreData(
                        "Looking for step:{} for mode {} and reached end of training. Max step available is {}".format(
                            step_num, mode, last_step
                        )
                    )
                raise StepNotYetAvailable(step_num, mode)
        assert False, "Should not happen"

    def values(self, mode=ModeKeys.GLOBAL, worker=None):
        res = {}
        for step in self.steps(mode=mode):
            res[step] = self.value(step_num=step, mode=mode, worker=worker)
        return res

    def value(self, step_num, mode=ModeKeys.GLOBAL, worker=None):
        # step refreshes
        s = self._step(step_num=step_num, mode=mode, worker=worker)
        if s.value is not None:
            return s.value
        elif s.location is not None:
            value = self.trial.index_reader.fetch_tensor_value(s.location)
            if self.cache:
                s.value = value
            return value
        else:
            has_reduction_values = len(s.reduction_values()) > 0
            has_reduction_locations = len(s.reduction_locations()) > 0
            has_reductions = has_reduction_locations or has_reduction_values
            raise TensorUnavailableForStep(self.name, step_num, mode, has_reductions)

    def shape(self, step_num, mode=ModeKeys.GLOBAL, worker=None):
        s = self._step(step_num=step_num, mode=mode, worker=worker)
        if s.shape is not None:
            return s.shape
        try:
            value = self.value(step_num, mode, worker)
            return value.shape
        except TensorUnavailableForStep:
            raise ShapeUnavailableForStep(self.name, step_num, mode)

    def reduction_values(self, step_num, mode=ModeKeys.GLOBAL, worker=None):
        s = self._step(step_num=step_num, mode=mode, worker=worker)
        if s is not None:
            rvs = {}
            if self.trial.index_mode:
                red_types = s.reduction_locations().keys()
            else:
                red_types = s.reduction_values().keys()
            for red_name, abs_val in red_types:
                rvs[(red_name, abs_val)] = self.reduction_value(
                    step_num, red_name, mode, worker, abs_val
                )
            return rvs
        else:
            assert False, "Should not happen"

    def workers(self, step_num, mode=ModeKeys.GLOBAL) -> list:
        step_dict = self._get_step_dict(step_num, mode)
        if step_dict is None:
            raise TensorUnavailableForStep(self.name, step_num, mode)
        return list(step_dict.keys())

    def reduction_value(
        self, step_num, reduction_name, mode=ModeKeys.GLOBAL, worker=None, abs=False
    ):
        """
        Returns the value of the reduction requested.
        If the tensor was saved as a reduction, then just fetches that.
        Else, tries to compute the reduction and returns. If the tensor value is not
        available, returns None as reduction
        Reductions are not cached. #TODO do we want to?
        :param step_num: step number
        :param mode: mode of job (train, eval, predict, etc).
                            If this is None, assumes step number is global
        :param reduction_name: name of reduction
        :param worker: name of worker
        :param abs: boolean which represents whether reduction should
                    be applied on absolute value of the tensor or not
        :return: reduction value requested as a float
        """

        s = self._step(step_num=step_num, mode=mode, worker=worker)
        rv = s.reduction_value(reduction_name, abs)
        rl = s.reduction_location(reduction_name, abs)
        if rv is not None:
            return rv
        elif rl is not None:
            return self.trial.index_reader.fetch_tensor_value(rl)
        else:
            if s.value is None and s.location is None:
                raise TensorUnavailableForStep(tname=reduction_name, step=step_num, mode=mode)
            elif s.value is None and s.location is not None:
                step_value = self.trial.index_reader.fetch_tensor_value(s.location)
                if self.cache:
                    s.value = step_value  # save value if cache is set to True
            else:
                step_value = s.value

            return get_numpy_reduction(reduction_name, step_value, abs)

    def _create_mode_step(self, mode, mode_step):
        mode_step = int(mode_step)
        if mode_step < 0:
            raise ValueError(
                "mode step number {} for tensor {} "
                "can not be less than 0".format(mode_step, self.name)
            )
        if mode not in self._mode_steps:
            self._mode_steps[mode] = ModeSteps(mode)

    def add_step(self, mode, mode_step, worker, tensor_location, tensor_shape):
        self._create_mode_step(mode, mode_step)
        if tensor_location is not None:
            self._mode_steps[mode].set_step_location(mode_step, worker, tensor_location)
        if tensor_shape is not None:
            self._mode_steps[mode].set_step_shape(mode_step, worker, tensor_shape.shape)
            self.original_name = tensor_shape.original_name

    def add_reduction_step(self, mode, mode_step, worker, red_name, abs, red_location):
        self._create_mode_step(mode, mode_step)
        self._mode_steps[mode].set_step_reduction_location(
            mode_step, worker, red_name, abs, red_location
        )

    def prev_steps(self, step, n=None, mode=ModeKeys.GLOBAL):
        """
        returns n prev steps from step representing step number
        of given mode including step
        :param step: int
            step number
        :param n: int
            number of previous steps to return
            if None returns all previous steps before step
        :param mode: value of the enum smdebug.modes
            modes.GLOBAL, modes.TRAIN, modes.EVAL, modes.PREDICT
        :return: a list of step numbers
        """
        steps = self.steps(mode=mode)
        i = bisect.bisect_right(steps, step)
        prev_steps = steps[:i]
        if n:
            return prev_steps[-n:]
        else:
            return prev_steps
