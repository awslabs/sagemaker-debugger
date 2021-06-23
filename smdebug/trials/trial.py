# Standard Library
import os
import time
from abc import ABC, abstractmethod
from bisect import bisect_left

# First Party
from smdebug.analysis.utils import _tensor_name_sorter, refresh
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.core.collection import Collection, CollectionKeys
from smdebug.core.config_constants import (
    INCOMPLETE_STEP_WAIT_WINDOW_DEFAULT,
    INCOMPLETE_STEP_WAIT_WINDOW_KEY,
    TRAINING_END_DELAY_REFRESH_DEFAULT,
    TRAINING_END_DELAY_REFRESH_KEY,
)
from smdebug.core.locations import IndexFileLocationUtils, TensorLocation, TensorShape
from smdebug.core.logger import get_logger
from smdebug.core.modes import ModeKeys
from smdebug.core.reductions import REDUCTIONS_PREFIX, reverse_reduction_tensor_name
from smdebug.core.tensor import StepState, Tensor
from smdebug.core.utils import (
    flatten,
    get_worker_name_from_collection_file,
    match_inc,
    serialize_tf_device,
)
from smdebug.exceptions import (
    MissingCollectionFiles,
    NoMoreData,
    StepUnavailable,
    TensorUnavailable,
)


class Trial(ABC):
    """
    The base class for creating an SMDebug trial objects.
    A trial creation helper function
    :class:`~smdebug.trials.create_trial` was introduced in
    the previous topic at :doc:`smdebug.trials`.

    After you create an SMDebug trial object, use the following
    Trial class methods for accessing output tensor information.

    """

    def __init__(
        self, name, range_steps=None, parallel=True, check=False, index_mode=True, cache=False
    ):
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
        self.index_reader = None
        self.index_tensors_dict = {}
        self.index_mode = index_mode
        self.last_event_token = None
        self.last_index_token = None
        self.worker_set = set()
        self.global_step_to_tensors_map = dict()
        self.mode_to_tensors_map = dict()
        self.num_workers = 0
        self.workers_for_global_step = {}
        self.last_complete_step = -1

        # INCOMPLETE_STEP_WAIT_WINDOW defines the maximum number
        # of incomplete steps that the trial will wait for before marking
        # half of them as complete.

        self._incomplete_wait_for_step_window = int(
            os.getenv(INCOMPLETE_STEP_WAIT_WINDOW_KEY, INCOMPLETE_STEP_WAIT_WINDOW_DEFAULT)
        )

        # this is turned off during rule invocation for performance reasons since
        # required tensors are already fetched
        self.dynamic_refresh = True
        # number of seconds to wait before refreshing after seeing end of trial
        self._training_end_delay_refresh = int(
            os.getenv(TRAINING_END_DELAY_REFRESH_KEY, TRAINING_END_DELAY_REFRESH_DEFAULT)
        )

        if self.range_steps is not None:
            assert self.range_steps[0] is None or (
                isinstance(self.range_steps[0], int) and self.range_steps[0] >= 0
            )
            assert self.range_steps[1] is None or (
                isinstance(self.range_steps[1], int) and self.range_steps[1] >= 0
            )
            if self.range_steps[1] is not None and self.range_steps[0] is not None:
                assert int(self.range_steps[1]) > int(self.range_steps[0]), (
                    "range_steps should be of the form " "(begin, end) where begin is less than end"
                )
            if self.range_steps[0] is not None and self.range_steps[1] is not None:
                self.logger.info(
                    "Trial {} will look for steps between {} and {}".format(
                        self.name, self.range_steps[0], self.range_steps[1]
                    )
                )

    def __repr__(self):
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>:(\n"
            f"    name={self.name},\n"
            f"    path={self.path},\n"
            f"    steps={self.steps()},\n"
            f"    collections={list(self.collections().keys())},\n"
            f"    tensor_names={self.tensor_names()},\n"
            f")"
        )

    @abstractmethod
    def _read_collections(self, collection_files):
        pass

    @abstractmethod
    def _get_collection_files(self):
        pass

    def _load_collections(self):
        num_times_before_warning = 10
        collection_files = []

        def _fetch():
            nonlocal collection_files
            nonlocal num_times_before_warning
            collection_files = self._get_collection_files()

            num_times_before_warning -= 1
            if num_times_before_warning < 0:
                self.logger.warning(
                    f"Waiting to read collections files generated by the training job,"
                    f"from {self.path}. "
                    f"If this has been a while, you might want to check that the "
                    f"trial is pointed at the right path."
                )
            else:
                self.logger.debug(
                    "Waiting to read collections files generated by the training job."
                )

        def _wait_for_collection_files(number_of_collection_file_to_wait_for):
            while len(collection_files) < number_of_collection_file_to_wait_for:
                time.sleep(2)
                _fetch()
                if has_training_ended(self.path):
                    """ _fetch should have returned all the collection files if the training job has ended """
                    if len(collection_files) < number_of_collection_file_to_wait_for:
                        raise MissingCollectionFiles

        _fetch()
        _wait_for_collection_files(1)  # wait for the first collection file
        self._read_collections(collection_files)
        _wait_for_collection_files(self.num_workers)  # wait for all the collection files
        for collection_file in collection_files:
            self.worker_set.add(get_worker_name_from_collection_file(collection_file))

    @abstractmethod
    def _load_tensors_from_index_tensors(self, index_tensors_dict):
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
        if training_ended and self.loaded_all_steps is False:
            retry_count = 2
        while retry_count > 0:
            if name is None:
                self.refresh_data()
            else:
                self.refresh_tensor(name)
            if retry_count > 1:
                self.logger.info(
                    f"Training has ended, will refresh one final time in "
                    f"{self._training_end_delay_refresh} sec."
                )
                time.sleep(self._training_end_delay_refresh)
            retry_count -= 1
        if training_ended is True and self.loaded_all_steps is False:
            self.loaded_all_steps = True
            self.last_complete_step = (
                sorted(self._global_to_mode.keys())[-1]
                if len(self._global_to_mode)
                else self.last_complete_step
            )  # Update last_complete_step to the last step written
            self.logger.info("Loaded all steps")
            self.logger.debug(
                f"Training Has Ended : last_complete_step was: {self.last_complete_step}"
            )
            self.logger.debug(f"Training Has Ended : last_index_token was: {self.last_index_token}")

    def refresh_tensor(self, tname, steps=None):
        # for now we load all tensors at once
        self.refresh_data()

    def tensor(self, tname):
        # will not show tensor if it was not written yet
        # has tensor will refresh
        """Retrieves the ``smdebug.core.tensor.Tensor`` object by the given name
        ``tname``. To find available methods that this Tensor object
        provides, see :doc:`tensor-api`.

        If output tensor is still not available when you run this method,
        it refreshes the method call until the first output tensor becomes available.

        Args:
            tname (str): Takes the name of tensor

        Returns:
            :class:`~smdebug.core.tensor.Tensor` object:
            An output tensor object.

        """

        if self.has_tensor(tname):
            return self._tensors[tname]
        else:
            raise TensorUnavailable(tname)

    def has_tensor(self, tname):
        # will return false if tensor was not written yet
        """Checks if the trial has a tensor of the given tensor name.

        Args:
            tname (str): Takes the name of tensor

        Returns:
            bool:
                ``True`` if the tensor is found by the trial, else it returns
                ``False``.

        """

        if tname not in self._tensors:
            self.maybe_refresh(tname)
        return tname in self._tensors

    def _populate_step_dict(self, mode, mode_step, step_num):
        if mode != ModeKeys.GLOBAL:
            if mode not in self._mode_to_global:
                self._mode_to_global[mode] = {}
            if mode_step not in self._mode_to_global[mode]:
                self._mode_to_global[mode][mode_step] = int(step_num)
        if step_num not in self._global_to_mode:
            self._global_to_mode[step_num] = (mode, mode_step)

    def _populate_workers_for_global_step(self, step, worker) -> None:
        """
        The self.workers_for_global_step dictionary holds a mapping of
        step number and a set of all the workers that have been written for the step.

        This function is used to add a worker to that set. To mark that a particular worker
        has finished writing the step.
        :param step:
        :param worker:
        :return: None
        """
        if step not in self.workers_for_global_step:
            self.workers_for_global_step[step] = set()
        self.workers_for_global_step[step].add(worker)
        self.logger.debug(f"Populated workers for global step:{step} worker: {worker}")

        if (
            len(self.workers_for_global_step[step]) == self.num_workers
            and step > self.last_complete_step
        ):
            self.last_complete_step = step
            self.logger.debug(f"Populating last completing step to: {step}")

    def _populate_global_step_to_tensor_name_map(self, tensorname: str, step_num) -> None:
        """
        The self.global_step_to_tensors_map dictionary holds a mapping of
        step number and a set of all the tensor names that have been written for the step.

        :param tensor:
        :param step_num:
        :return: None
        """
        if step_num not in self.global_step_to_tensors_map:
            self.global_step_to_tensors_map[step_num] = set()
        self.global_step_to_tensors_map[step_num].add(tensorname)

    def _populate_mode_to_tensor_name_map(self, tensorname, mode) -> None:
        """
        The self.mode_to_tensors_map dictionary holds a mapping of
        mode and a set of all the tensor names that have been written for the mode.
        :param tensor:
        :return:
        """
        if mode != ModeKeys.GLOBAL:
            if mode not in self.mode_to_tensors_map:
                self.mode_to_tensors_map[mode] = set()
            self.mode_to_tensors_map[mode].add(tensorname)

    def _load_tensors_from_index_tensors(self, index_tensors_dict):
        for tname in index_tensors_dict:
            for step, itds in index_tensors_dict[tname].items():
                for worker in itds:
                    self._add_tensor(
                        int(step),
                        worker,
                        itds[worker].get("tensor_location", None),
                        itds[worker].get("tensor_shape", None),
                    )

    def _add_tensor(
        self, step_num, worker, tensor_location: TensorLocation, tensor_shape: TensorShape
    ):
        is_reduction = False

        if tensor_location is not None:
            tensorname = tensor_location.tensorname
            mode = tensor_location.mode
            mode_step = tensor_location.mode_step
        elif tensor_shape is not None:
            tensorname = tensor_shape.name
            mode = tensor_shape.mode
            mode_step = tensor_shape.mode_step
        else:
            raise RuntimeError("both tensor_location and tensor_shape can't be None")

        if REDUCTIONS_PREFIX in tensorname:
            tensorname, red_name, abs = reverse_reduction_tensor_name(tensorname)
            is_reduction = True

        if tensorname not in self._tensors:
            tensor = Tensor(tensorname, trial=self, cache=self.cache)
            self._tensors[tensorname] = tensor

        tensor = self._tensors[tensorname]

        if is_reduction:
            tensor.add_reduction_step(mode, mode_step, worker, red_name, abs, tensor_location)
        else:
            # shape can only be passed for actual tensor, not reductions
            tensor.add_step(mode, mode_step, worker, tensor_location, tensor_shape)

        self._populate_step_dict(mode, mode_step, step_num)
        self._populate_global_step_to_tensor_name_map(tensorname, step_num)
        self._populate_workers_for_global_step(step_num, worker)
        self._populate_mode_to_tensor_name_map(tensorname, mode)

    def _tensors_matching_regex(self, regex_list) -> set:
        matched_tensornames = set()
        if not isinstance(regex_list, list):
            regex_list = [regex_list]
        regex_list = flatten(regex_list)
        for tensorname in self._tensors.keys():
            if match_inc(tensorname, regex_list):
                matched_tensornames.add(tensorname)
        return matched_tensornames

    @staticmethod
    def _parse_collection_name(collection):
        if isinstance(collection, Collection):
            coll_name = collection.name
        elif type(collection) is str:
            coll_name = collection
        else:
            raise TypeError(f"Invalid argument type for collection {collection.__class__}")
        return coll_name

    def _tensors_in_collection(self, collection) -> set:
        coll_name = self._parse_collection_name(collection)
        rval = set(self.collection(coll_name).tensor_names)
        regex = self.collection(coll_name).include_regex
        if regex:
            rval.update(self._tensors_matching_regex(regex))
        return rval

    def inputs(self, step, mode=ModeKeys.GLOBAL):
        input_tensors_names = sorted(
            self.tensor_names(
                show_prefixed_tensors=True, step=step, mode=mode, collection=CollectionKeys.INPUTS
            ),
            key=_tensor_name_sorter,
        )
        input_tensors = [
            self.tensor(tensor_name).value(step) for tensor_name in input_tensors_names
        ]
        return input_tensors

    def _get_output_tensors_helper(self, step, mode, regex):
        output_tensors_names = sorted(
            self.tensor_names(show_prefixed_tensors=True, step=step, mode=mode, regex=regex),
            key=_tensor_name_sorter,
        )
        output_tensors = [
            self.tensor(tensor_name).value(step) for tensor_name in output_tensors_names
        ]
        return output_tensors

    def labels(self, step, mode=ModeKeys.GLOBAL):
        return self._get_output_tensors_helper(step, mode, regex="labels*")

    def predictions(self, step, mode=ModeKeys.GLOBAL):
        return self._get_output_tensors_helper(step, mode, regex="predictions*")

    # * is used in python to force usage of named arguments
    def tensor_names(
        self,
        show_prefixed_tensors=False,
        *,
        step=None,
        mode=ModeKeys.GLOBAL,
        regex=None,
        collection=None,
    ) -> list:
        """Retrieves names of tensors saved.

        All arguments to this method are optional. You are not required to pass
        any of the following arguments. By default, this method returns all tensor names
        if you don't pass any arguments.

        Args:
            step (int): If you want to retrieve the list of tensors saved at a
                particular step, pass the step number as an integer. This step number
                will be treated as step number corresponding to the mode passed
                below. By default it is treated as global step.

            mode (smdebug.modes enum value): If you want to retrieve the list
                of tensors saved for a particular mode, pass the mode here as
                ``smd.modes.TRAIN``, ``smd.modes.EVAL``, ``smd.modes.PREDICT``, or
                ``smd.modes.GLOBAL``.

            regex (str or list[str]): You can filter tensors matching regex
                expressions by passing a regex expressions as a string or list of
                strings. You can only pass one of ``regex`` or ``collection``
                parameters.

            collection (Collection or str): You can filter tensors belonging
                to a collection by either passing a collection object or the name of
                collection as a string. You can only pass one of ``regex`` or
                ``collection`` parameters.

        Returns:
            list[str]:
                List of strings representing names of tensors matching
                the given arguments. Arguments are processed as follows: get the list of
                tensor names for given step and mode, saved for given step matching all
                the given arguments, i.e. intersection of tensors matching each of the
                parameters.

        **Examples:**

        - ``trial.tensor_names()`` - Returns all tensors saved for any step or
          mode.
        - ``trial.tensor_names(step=10, mode=modes.TRAIN)`` - Returns tensors
          saved for training step 10
        - ``trial.tensor_names(regex='relu')`` - Returns all tensors matching the
          regex pattern ``relu`` saved for any step or mode.
        - ``trial.tensor_names(collection='gradients')`` - Returns tensors from
          collection “gradients”
        - ``trial.tensor_names(step=10, mode=modes.TRAIN, regex='softmax')`` -
          Returns tensor saved for 10th training step which matches the regex
          ``softmax``

        """
        self.maybe_refresh()
        ts = set()
        if step is None and mode == ModeKeys.GLOBAL:
            ts.update(self._tensors.keys())
        if step is None and mode != ModeKeys.GLOBAL:
            ts.update(self.mode_to_tensors_map[mode])
        else:
            ts.update(self._tensors_for_step(step, mode))
        self.logger.debug(
            f"getting tensor_names with params: step:{step} mode:{mode} regex:{regex} collection:{collection}"
        )

        if regex is None and collection is None:
            return sorted(list(ts))
        elif regex is not None and collection is not None:
            raise ValueError("Only one of `regex` or `collection` can be passed to this method")
        else:
            if collection is not None:
                xs = set(self._tensors.keys()).intersection(self._tensors_in_collection(collection))
                matching_tensors_saved = ts.intersection(xs)
                if len(matching_tensors_saved) == 0:
                    coll_name = self._parse_collection_name(collection)
                    self.logger.warning(f"No tensors from the collection {coll_name} were saved")
            else:
                xs = self._tensors_matching_regex(regex)
                matching_tensors_saved = ts.intersection(xs)
                if len(matching_tensors_saved) == 0:
                    self.logger.warning(
                        f"No tensors matching the regex pattern:{regex} given were saved"
                    )
            return sorted(list(matching_tensors_saved))

    def _tensors_for_step(self, step, mode=ModeKeys.GLOBAL) -> list:
        step = self._mode_to_global[mode][step] if mode != ModeKeys.GLOBAL else step
        if step in self.global_step_to_tensors_map:
            return list(self.global_step_to_tensors_map[step])
        return []

    def workers(self):
        """Query for all the worker processes from which data was saved by smdebug
        during multi worker training.

        Returns:
            list[str]:
                A sorted list of names of worker processes from which data
                was saved. If using TensorFlow Mirrored Strategy for multi worker
                training, these represent names of different devices in the process. For
                Horovod, torch.distributed and similar distributed training approaches,
                these represent names of the form ``worker_0`` where 0 is the rank of
                the process.

        """

        self.maybe_refresh()
        return sorted(list(self.worker_set))

    def steps(self, mode=ModeKeys.GLOBAL, show_incomplete_steps=False) -> list:
        """Retrieves a list of steps collected by SageMaker Debugger.

        Args:
            mode (smdebug.modes enum value): Passing a mode here allows you
                want to retrieve the list of steps seen by a trial for that mode If
                this is not passed, returns steps for all modes.
            show_incomplete_steps (bool)

        Returns:
            list[int]:
                List of integers representing step numbers. If a mode was
                passed, this returns steps within that mode, i.e. mode steps. Each of
                these mode steps has a global step number associated with it. The global
                step represents the sequence of steps across all modes executed by the
                job.

        """
        all_steps = self._all_steps(mode)
        if show_incomplete_steps is True:
            return all_steps
        completed_steps = list()
        for step in all_steps:
            global_step = self._mode_to_global[mode][step] if mode != ModeKeys.GLOBAL else step
            if (
                len(self.workers_for_global_step[global_step]) == self.num_workers
                or self.loaded_all_steps is True
                or self.last_complete_step >= global_step
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
        elif mode in self._mode_to_global and mode_step in self._mode_to_global[mode]:
            return self._mode_to_global[mode][mode_step]

    def global_step(self, mode, mode_step):
        """Given a mode and a mode_step number you can retrieve its global step
        using this method.

        Args:
            mode (smdebug.modes enum value): Takes the mode as enum value
            mode_step (int): Takes the mode step as an integer

        Returns:
            ``int``:
                An integer representing ``global_step`` of the given mode and
                ``mode_step``.

        """

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
        """Given a global step number you can identify the ``mode_step`` for that
        step using this method.

        Args:
            global_step (int): Takes the global step as an integer.

        Returns:
            ``int``:
                An integer representing ``mode_step`` of the given global step.
                Typically used in conjunction with ``mode`` method.

        """

        x = self.mode_modestep(global_step)
        if x:
            return x[1]

    def mode(self, global_step):
        # can return global mode in some cases
        """Given a global step number you can identify the mode for that step using
        this method.

        Args:
            global_step (int): Takes the global step as an integer.

        Returns:
            ``smdebug.modes enum value`` of the given global step.

        """

        x = self.mode_modestep(global_step)
        if x:
            return x[0]

    def modes(self):
        # will not return global mode
        """Retrieve a list of modes seen by the trial.

        Returns:
            list[smdebug.modes enum value]:
                List of modes for which data was
                saved at all steps collected from the training job.

        """

        return self._mode_to_global.keys()

    def collections(self):
        """List the collections from the trial.

        Note that tensors part of these
        collections may not necessarily have been saved from the training job.
        Whether a collection was saved or not depends on the configuration of
        the Hook during training.

        Returns:
            dict[str -> Collection]:
                A dictionary indexed by the name of the
                collection, with the Collection object as the value. Please refer
                :doc:`collections` for more details.

        """

        return self.collection_manager.collections

    def collection(self, coll_name):
        """Get a specific collection from the trial.

        Note that tensors which are
        part of this collection may not necessarily have been saved from the
        training job. Whether this collection was saved or not depends on the
        configuration of the Hook during training.

        Args:
            coll_name (str): Name of the collection

        Returns:
            ``Collection``:
                The requested Collection object. Please refer
                :doc:`collections` for more details.

        """
        return self.collection_manager.get(coll_name)

    def wait_for_steps(self, required_steps, mode=ModeKeys.GLOBAL):
        """This method allows you to wait for steps before proceeding.

        You might
        want to use this method if you want to wait for smdebug to see the
        required steps so you can then query and analyze the tensors saved by
        that step. This method blocks till all data from the steps are seen by
        smdebug.

        Args:
            required_steps (list[int]): Step numbers to wait for
            mode (smdebug.modes enum value): The mode to which given step
                numbers correspond to. This defaults to modes.GLOBAL.

        Returns:
            None:
                Only returns after we know definitely whether we have seen
                the steps.

        **Exceptions raised:**

        ``StepUnavailable`` and ``NoMoreData``. See `Exceptions <#exceptions>`__
        section for more details.

        """

        with refresh(self):
            for step in required_steps:
                while True:
                    s = self.has_passed_step(step, mode)
                    if s == StepState.AVAILABLE:
                        break
                    elif s == StepState.UNAVAILABLE:
                        if self.loaded_all_steps is False:
                            raise StepUnavailable(step, mode)
                        else:
                            last_step = -1
                            avail_steps = self._all_steps(mode=mode)
                            if len(avail_steps) > 0:
                                last_step = avail_steps[-1]
                            if step < last_step:
                                raise StepUnavailable(step, mode)
                            raise NoMoreData(step, mode, last_step)
                    time.sleep(2)

    def has_passed_step(self, step, mode=ModeKeys.GLOBAL) -> StepState:
        """
        This function indicates whether a step is complete (AVAILABLE),
        incomplete ( NOT_YET_AVAILABLE ) or absent ( UNAVAILABLE ).

        **Overview of logic:**

        1. if the queried step is greater than all the available steps (complete / incomplete):

            .. code:: python

                if job is not complete:
                    return StepState.NOT_YET_AVAILABLE
                else:
                    return StepState.UNAVAILABLE

        2. if the queried step is less or equal to a step in available steps (complete / incomplete):

            .. code:: python

                if the queried step is less than all the available steps:
                    if single_worker:
                        return UNAVAILABLE ( step has been skipped or will not written)
                    else:
                        return NOT_YET_AVAILABLE

        3. queried step is available:

            .. code:: python

                if all workers have written the step or job is complete
                or last_complete_step > step ( All workers have written a step greater than the step we are checking.
                                                    Hence, the step will never be complete. )
                    return AVAILABLE
                else:
                     return NOT_YET_AVAILABLE

        Args:
            step (int): The step number to check if the trial has passed it.
            mode (smdebug.modes enum value): The mode to which given step
                number corresponds to. This defaults to modes.GLOBAL.

        Returns:
            smdebug.core.tensor.StepState enum value:
                Returns one of the following
                values: ``UNAVAILABLE``, ``AVAILABLE``,
                and ``NOT_YET_AVAILABLE``.

        """
        all_steps = self.steps(mode=mode, show_incomplete_steps=True)
        bisect_idx = bisect_left(all_steps, step)

        if bisect_idx < len(all_steps):
            # This returns either the global step corresponding to the mode-step
            # or the closest global step that is greater than the step passed as a parameter
            g_step = self._global_step_currently(mode, all_steps[bisect_idx])
            if all_steps[bisect_idx] > step:
                if self.last_complete_step >= g_step:
                    return StepState.UNAVAILABLE
                return StepState.NOT_YET_AVAILABLE
            elif all_steps[bisect_idx] == step:
                if len(self.workers_for_global_step[g_step]) == self.num_workers:
                    return StepState.AVAILABLE
                elif self.loaded_all_steps is True:
                    self.logger.info(
                        f"Step {step} of mode {mode} was written only by workers: {self.workers_for_global_step[step]}"
                    )
                    self.logger.info(
                        f"Step {step} of mode {mode} was marked complete because the job is complete"
                    )
                    return StepState.AVAILABLE
                elif g_step <= self.last_complete_step:
                    self.logger.info(
                        f"Step {step} of mode {mode} was written only by workers: {self.workers_for_global_step[g_step]}"
                    )
                    self.logger.info(
                        f"Step {step} of mode {mode} was marked complete because the last complete step is {self.last_complete_step}"
                    )
                    return StepState.AVAILABLE
                else:
                    return StepState.NOT_YET_AVAILABLE
        if self.loaded_all_steps is True:
            return StepState.UNAVAILABLE
        return StepState.NOT_YET_AVAILABLE

    def _update_last_index_token(self, new_index_token: str) -> None:
        """
        This function updates the last_index_token in the following scenarios:
            1. last_complete_step >= last_index_token_step :
                this means that the token isn't pointing to the latest completed step
            2. number of steps available ( complete or incomplete ) - (last_completed_step+1) > window_size_limit:
                we maintain a window to stop querying for older steps that have not completed.
                if the total number of steps, we are querying for completion is greater than our window_size_limit
                we update the last_index_token and last_complete_step by (window_size_limit // 2)
        :param new_index_token:
        :return:None

        """
        if self.last_index_token is None:
            last_index_token_step = 0
        else:
            last_index_token_step = IndexFileLocationUtils.parse_step_from_index_file_name(
                self.last_index_token
            )

        # Case 1: This case is not satisfied when all workers in a
        # distributed training job have not written a step
        if self.last_complete_step >= last_index_token_step:
            prefix = IndexFileLocationUtils.get_prefix_from_index_file(new_index_token)
            # sort lexicographically and select the last worker
            last_worker = sorted(list(self.worker_set))[-1]
            # below converts worker_name to serialized workerName
            # if it's a tf device, else no effect
            last_worker_serialized = serialize_tf_device(last_worker)
            self.last_index_token = IndexFileLocationUtils.get_index_key_for_step(
                prefix, self.last_complete_step, last_worker_serialized
            )
            self.logger.debug(f"Updated last index token to:{self.last_index_token}")

        # Case 2: This case is satisfied if the number of incomplete steps
        # is greater than the INCOMPLETE_STEP_WAIT_WINDOW
        available_step = self._global_to_mode.keys()
        if (
            len(available_step) - (self.last_complete_step + 1)
            > self._incomplete_wait_for_step_window
        ):
            prefix = IndexFileLocationUtils.get_prefix_from_index_file(new_index_token)
            last_worker = sorted(list(self.worker_set))[-1]
            # below converts worker_name to serialized workerName
            # if it's a tf device, else no effect
            last_worker_serialized = serialize_tf_device(last_worker)
            self.last_index_token = IndexFileLocationUtils.get_index_key_for_step(
                prefix,
                self.last_complete_step + (self._incomplete_wait_for_step_window // 2),
                last_worker_serialized,
            )
            self.last_complete_step = IndexFileLocationUtils.parse_step_from_index_file_name(
                self.last_index_token
            )
            self.logger.info(
                f"Waiting for: {len(available_step) - (self.last_complete_step + 1)} Steps. \n"
                f"INCOMPLETE_STEP_WAIT_WINDOW: {self._incomplete_wait_for_step_window}. \n"
                f"Marking the last {self._incomplete_wait_for_step_window // 2} incomplete steps as complete"
                f"Updating last_index_token to: {self.last_index_token}. \n"
                f"Updating last_complete_step to: {self.last_complete_step}. "
            )

    def refresh_data(self):
        # TODO if job finished
        if self.index_mode:
            index_tensors_dict, new_index_token = self.index_reader.load_tensor_data_from_index_files(
                start_after_key=self.last_index_token, range_steps=self.range_steps
            )
            if len(index_tensors_dict):
                self.index_tensors_dict.update(index_tensors_dict)
                self._load_tensors_from_index_tensors(index_tensors_dict)
            if new_index_token:  # new index token can be None if there are no new index files
                self._update_last_index_token(new_index_token)
