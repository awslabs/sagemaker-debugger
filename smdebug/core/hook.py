# Standard Library
import atexit
import os
import re as _re
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

# Third Party
import numpy as np

# First Party
from smdebug.core.access_layer import training_has_ended
from smdebug.core.collection import (
    NON_HISTOGRAM_COLLECTIONS,
    NON_REDUCTION_COLLECTIONS,
    SCALAR_COLLECTIONS,
    SM_METRIC_COLLECTIONS,
    Collection,
    CollectionKeys,
)
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import (
    DEFAULT_WORKER_NAME,
    LATEST_GLOBAL_STEP_SAVED,
    LATEST_GLOBAL_STEP_SEEN,
    LATEST_MODE_STEP,
    TRAINING_RUN,
)
from smdebug.core.hook_utils import get_tensorboard_dir, verify_and_get_out_dir
from smdebug.core.json_config import create_hook_from_json_config
from smdebug.core.logger import get_logger
from smdebug.core.modes import ALLOWED_MODE_NAMES, ALLOWED_MODES, ModeKeys
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.reductions import get_reduction_tensor_name
from smdebug.core.sagemaker_utils import is_sagemaker_job
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.core.state_store import StateStore
from smdebug.core.utils import (
    flatten,
    get_tb_worker,
    is_first_process,
    match_inc,
    remove_claim_file,
    size_and_shape,
)
from smdebug.core.writer import FileWriter
from smdebug.exceptions import InvalidCollectionConfiguration

try:
    from smexperiments.metrics import SageMakerFileMetricsWriter
except ImportError:
    SageMakerFileMetricsWriter = None


logger = get_logger()


class ScalarCache(object):
    def __init__(
        self, scalar_name, scalar_val, mode, sm_metric, write_tb, write_event, timestamp=None
    ):
        """

        Args:
            scalar_name: Name of the scalar to be stored
            scalar_val: Value of scalar
            mode: Modekey
            sm_metric: True or False indicates whether the scalar will be written to SageMaker
            write_tb: True or False indicates whether scalar will be written to Tensorboard
            write_event: True or False indicates whether scalar will be writen to event file.
            timestamp: Timestamp at which this object is created.
        The 'save_scalar()' method creates objects of this class and caches the scalars that users intends to store.
        These objects will be written to disk in the next available step.
        """
        self.name = scalar_name
        self.value = scalar_val
        self.mode = mode
        self.sm_metric = sm_metric
        self.write_tb = write_tb
        self.write_event = write_event
        self.timestamp = timestamp if timestamp else time.time()


class BaseHook:
    __metaclass__ = ABCMeta

    def __init__(
        self,
        collection_manager: CollectionManager,
        default_include_collections: List[str],
        init_step: int = 0,
        out_dir: Optional[str] = None,
        export_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
        dry_run: bool = False,
        reduction_config: Optional[ReductionConfig] = None,
        save_config: Optional[Union[SaveConfig, Dict[ModeKeys, SaveConfigMode]]] = None,
        include_regex: Optional[List[str]] = None,
        include_collections: Optional[List[str]] = None,
        save_all: bool = False,
        include_workers: str = "one",
    ):
        """
        A class used to represent the hook which gets attached to the
        training process. This takes the form appropriate for the framework
        such as tf.train.SessionRunHook for TF, Callback for keras...

        ...

        Attributes
        ----------
        out_dir : str
            represents a path into which outputs will be written to. The hook raises error if the 'out_dir' already
            exists. The implementation does not support merging the tensors generated in current job with tensors
            from previous job. Hence, ensure that the 'out_dir' does not exist.
        dry_run : bool
            when dry run is set, behavior is only described in the log file.
            tensors are not actually saved.
        save_config: SaveConfig object
            Takes save config object which is applied as default for all included tensors.
            A collection can optionally have its own saveconfig object
            which overrides this for its tensors.

        reduction_config: ReductionConfig object
            if passed, this reduction config object is used
            as default for all tensors included.
            A collection has its own saveconfig object
            which overrides this for its tensors. if this is not passed,
            tensor is saved in full.

        include_regex: list of str
            takes as input the list of string representing regular expressions. Tensors whose names match
            these regular expressions will be saved. These tensors will be available as part of the `default`
            collection.

        include_collections: list of str representing collection names
            takes as input the collections which should be saved.
            if this is empty, it defaults to including all collections from code

        save_all: bool
            a shortcut for saving all tensors in the model.
            they are all saved in the collection `all`
        include_workers: str
            makes the hook save data from all workers
        """
        self.out_dir = verify_and_get_out_dir(out_dir)
        self.tensorboard_dir = get_tensorboard_dir(
            export_tensorboard=export_tensorboard,
            tensorboard_dir=tensorboard_dir,
            out_dir=self.out_dir,
        )

        self.dry_run = dry_run
        self.worker = None
        # when smdebug is used during an unsupported dist training process
        # we write data only from the process that has self.first_process set to True.
        self.first_process = None
        self.save_all_workers = True if include_workers == "all" else False
        self.chief_worker = DEFAULT_WORKER_NAME

        if include_collections is None:
            include_collections = default_include_collections
        else:
            include_collections = flatten(include_collections)
        self.include_collections = list(
            set(include_collections).union(set(default_include_collections))
        )

        self.save_all = save_all
        self.save_config = SaveConfig.parse(save_config)
        if reduction_config is None:
            reduction_config = ReductionConfig(save_raw_tensor=True)
        self.reduction_config = reduction_config
        self.include_regex = include_regex
        self.collection_manager = collection_manager
        self.init_step = init_step

        # The written_tensor_name_for_step dictionary stores
        # the names of each tensor saved for every step.
        # This is to detect name clashes.
        # If a name clash is detected, it is avoided by appending
        # an index to the tensor name.
        self.written_tensor_name_for_step = defaultdict(int)

        self.logger = logger

        if self.tensorboard_dir is None:
            self.logger.info(
                f"tensorboard_dir has not been set for the hook. SMDebug will not be exporting tensorboard summaries."
            )

        if include_regex is not None:
            collection_manager.get(CollectionKeys.DEFAULT).include(include_regex)
            if CollectionKeys.DEFAULT not in self.include_collections:
                self.include_collections.append(CollectionKeys.DEFAULT)

        self.save_all = save_all
        if self.save_all:
            collection_manager.get(CollectionKeys.ALL).include(".*")
            if CollectionKeys.ALL not in self.include_collections:
                self.include_collections.append(CollectionKeys.ALL)

        if (
            CollectionKeys.DEFAULT not in self.include_collections
            and collection_manager.get(CollectionKeys.DEFAULT).include_regex
        ):
            self.logger.warn(
                "The `default` collection was not passed to "
                "include_collections. So it is not being saved"
            )

        self._collections_to_save = set()
        self._collections_to_save_for_step = None
        self.prepared_collections = False
        self.tensor_to_collections = {}

        self.step = init_step
        self.last_saved_step = None
        self.mode = ModeKeys.GLOBAL
        self.mode_steps = {ModeKeys.GLOBAL: init_step}
        self.writer = None

        if is_sagemaker_job() and SageMakerFileMetricsWriter is not None:
            self.metrics_writer = SageMakerFileMetricsWriter()
        else:
            self.metrics_writer = None

        # Maps ModeKeys to FileWriter objects
        self.tb_writers = {}

        # Cache scalars that are being saved through save_scalar() calls
        self.scalar_cache = []

        self.logger.info("Saving to {}".format(self.out_dir))
        atexit.register(self._cleanup)

        # Check if there is any last saved state. Initialize the hook based last saved state.
        self.training_run = 0
        self._initialize_to_last_saved_state()

    def _initialize_to_last_saved_state(self):
        self.state_store = StateStore()
        last_state = self.state_store.get_last_saved_state()
        if last_state is not None:
            self.last_saved_step = last_state[LATEST_GLOBAL_STEP_SAVED]
            self.init_step = last_state[LATEST_GLOBAL_STEP_SEEN]
            self.training_run = 1 + last_state[TRAINING_RUN]
            for (mode, step) in last_state[LATEST_MODE_STEP].items():
                self.mode_steps[ModeKeys[mode]] = step
            self.mode_steps[ModeKeys.GLOBAL] = self.init_step
            self.step = self.init_step
            self.logger.info(
                f"Initialized the hook with the last saved state: last_saved_step={self.last_saved_step} init_step = {self.init_step}, step = {self.step} mode_steps = {str(self.mode_steps)}"
            )

    def __repr__(self):
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>:(\n"
            f"    out_dir={self.out_dir},\n"
            f"    tensorboard_dir={self.tensorboard_dir},\n"
            f"    step={self.step},\n"
            f"    mode={self.mode},\n"
            f"    mode_steps={self.mode_steps},\n"
            f"    include_collections={self.include_collections},\n"
            f"    writer={self.writer},\n"
            f"    save_config={str(self.save_config)[:200]} ...>,\n"
            f"    reduction_config={str(self.reduction_config)},\n"
            f"    save_all={self.save_all},\n"
            f"    dry_run={self.dry_run},\n"
            f")"
        )

    @classmethod
    def create_from_json_file(cls, json_file_path=None):
        """Relies on the existence of a JSON file.

        First, check json_config_path. If it's not None,
            If the file exists, use that.
            If the file does not exist, throw an error.
        Otherwise, check the filepath set by a SageMaker environment variable.
            If the file exists, use that.
        Otherwise,
            return None.
        """
        return create_hook_from_json_config(cls, json_config_path=json_file_path)

    @abstractmethod
    def _get_worker_name(self):
        pass

    @abstractmethod
    def _get_num_workers(self):
        pass

    @abstractmethod
    def _is_not_supported(self):
        pass

    #### Save Manager methods ####

    def _should_collection_be_saved(self, coll_name: str) -> bool:
        return coll_name in self.include_collections

    def _assert_prep(self):
        assert self.prepared_collections, "Collections have not been prepared yet"

    def _get_all_collections_to_save(self) -> Set["Collection"]:
        self._assert_prep()
        return self._collections_to_save

    def _is_collection_being_saved_for_step(self, name):
        # if saving all, all collections will be part of colls_for_step
        colls_for_step = self._get_collections_to_save_for_step()
        return self.collection_manager.get(name) in colls_for_step

    def _get_collections_to_save_for_step(self) -> Set["Collection"]:
        if self._collections_to_save_for_step is None:
            self._assert_prep()
            self._collections_to_save_for_step = set()
            for coll in self._get_all_collections_to_save():
                if self.mode in [ModeKeys.EVAL, ModeKeys.PREDICT]:
                    if coll.name in [CollectionKeys.GRADIENTS, CollectionKeys.OPTIMIZER_VARIABLES]:
                        continue
                if coll.save_config.should_save_step(self.mode, self.mode_steps[self.mode]):
                    self._collections_to_save_for_step.add(coll)

            if self._collections_to_save_for_step:
                if self.mode == ModeKeys.GLOBAL:
                    step_str = f"for step {self.step}"
                else:
                    step_str = f"for step {self.mode_steps[self.mode]} of mode {self.mode.name}"
                self.logger.debug(
                    f"Saving the collections "
                    f"{', '.join([x.name for x in self._collections_to_save_for_step])} {step_str}"
                )
        return self._collections_to_save_for_step

    def _get_collections_with_tensor(self, tensor_name) -> Set["Collection"]:
        self._assert_prep()
        # for tf this will be prepopulated in check_and_add_tensor
        if tensor_name not in self.tensor_to_collections:
            # for mxnet it is computed and then cached
            matched_colls = set()
            for coll in self._collections_to_save:
                if tensor_name in coll.tensor_names:
                    # if being matched as reduction,
                    # it must be in reduction_tensor_name, not with regex
                    matched_colls.add(coll)
                elif match_inc(tensor_name, coll.include_regex):
                    coll.add_tensor_name(tensor_name)
                    matched_colls.add(coll)
            self.tensor_to_collections[tensor_name] = matched_colls
        return self.tensor_to_collections[tensor_name]

    @abstractmethod
    def _get_default_collections(self):
        pass

    def _prepare_collections(self):
        """Populate collections_to_save and ensure every collection has
        a save_config and reduction_config."""
        for c_name, c in self.collection_manager.get_collections().items():
            if c_name not in self._get_default_collections():
                if bool(c.include_regex) is False and bool(c.tensor_names) is False:
                    raise InvalidCollectionConfiguration(c_name)
            if c in self._collections_to_save:
                continue
            elif self._should_collection_be_saved(CollectionKeys.ALL):
                self._collections_to_save.add(c)
            elif self._should_collection_be_saved(c_name):
                self._collections_to_save.add(c)

        self.logger.info(
            f'Monitoring the collections: {", ".join([x.name for x in self._collections_to_save])}'
        )
        # Populate configs_for_collections and reduction_config
        for c_name, c in self.collection_manager.get_collections().items():

            if c_name in NON_HISTOGRAM_COLLECTIONS:
                c.save_histogram = False

            if c.save_config is None:
                # Set to the default if None
                c.save_config = self.save_config
            elif isinstance(c.save_config, SaveConfig):
                # Otherwise, set missing modes to the defaults
                c.save_config.merge_default_save_config(self.save_config)
            else:
                raise TypeError(f"save_config={c.save_config} must be None or SaveConfig")

            if c_name in NON_REDUCTION_COLLECTIONS:
                c.reduction_config = ReductionConfig(save_raw_tensor=True)
            elif c.reduction_config is None:
                c.reduction_config = self.reduction_config

        self.prepared_collections = True

    #### End of Save Manager methods ####

    def _close_writers(self) -> None:
        if self.dry_run:
            return

        # flush out sm_metric scalars to metrics file
        self._write_scalars()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        to_delete_writers = []

        # Delete all the tb writers
        for mode, writer in self.tb_writers.items():
            if writer is not None:
                writer.flush()
                writer.close()
                to_delete_writers.append(mode)
        for mode in to_delete_writers:
            del self.tb_writers[mode]

    def _initialize_writers(self, only_initialize_if_missing=False) -> None:
        # Function is overridden in smdebug/tensorflow/base_hook.py
        if only_initialize_if_missing and self.writer:
            return
        if self.dry_run:
            return
        if self.first_process is False:
            return
        elif self.first_process is None:
            if self._get_num_workers() == 1:
                if is_first_process(self.out_dir):
                    self.first_process = True
                    self.logger.info(f"Hook is writing from the hook with pid: {os.getpid()}\n")
                else:
                    if self.first_process is None:
                        self.logger.warn(
                            f"Unsupported Distributed Training Strategy Detected. \
                            Sagemaker-Debugger will only write from one process. \
                            The process with pid: {os.getpid()} will not be writing any data. \n"
                        )
                    self.first_process = False
                    return

        if self.save_all_workers is False:
            if self.worker != self.chief_worker:
                return
        self.writer = FileWriter(trial_dir=self.out_dir, step=self.step, worker=self.worker)

    def _get_writers(self, tensor_name, tensor_ref=None) -> List[FileWriter]:
        """
        :param tensor_name:
        :param tensor_ref: used by TF
        :return: List[FileWriter]
        """
        if self.save_all_workers is False and self.worker != self.chief_worker:
            return []
        return [self.writer] if self.writer else []

    def _maybe_get_tb_writer(self) -> Optional[FileWriter]:
        """ Returns a FileWriter object if `hook.tensorboard_dir` has been specified, else None.

        Creates a writer if does not exist.
        """
        if not self.tensorboard_dir:
            return None

        if self.mode in self.tb_writers:
            assert self.tb_writers[self.mode] is not None
            # would be there if set_mode was called
            return self.tb_writers[self.mode]
        else:
            # s = self.step
            # if s < 0: s = 0
            self.tb_writers[self.mode] = FileWriter(
                trial_dir=self.tensorboard_dir,
                step=self.step,
                worker=get_tb_worker(),
                write_checksum=True,
                wtype="tensorboard",
                mode=self.mode,
            )
            return self.tb_writers[self.mode]

    def _close_tb_writer(self):
        if self.dry_run:
            return

        if self.mode in self.tb_writers:
            self.tb_writers[self.mode].close()
            del self.tb_writers[self.mode]

    def close(self):
        self._cleanup()

    def _cleanup(self):
        self._close_writers()

        if self.metrics_writer:
            self.metrics_writer.close()

        training_has_ended(self.out_dir)
        if self.first_process is True:
            remove_claim_file(self.out_dir)

    def _increment_step(self):
        # Update the last_state to the last step number that was saved or seen
        self._write_state()

        self.step += 1
        self.mode_steps[self.mode] += 1
        self.written_tensor_name_for_step.clear()

        # Increment Global step number irrespective of what mode it is
        if self.mode != ModeKeys.GLOBAL:
            self.mode_steps[ModeKeys.GLOBAL] = self.step
        self._collections_to_save_for_step = None

    def _write_state(self):
        if self.state_store.is_checkpoint_updated():
            current_state = dict()
            current_state[TRAINING_RUN] = self.training_run
            current_state[LATEST_GLOBAL_STEP_SAVED] = self.last_saved_step
            current_state[LATEST_GLOBAL_STEP_SEEN] = self.step
            mode_step = dict()
            for (mode, step) in self.mode_steps.items():
                mode_step[mode.name] = step
            current_state[LATEST_MODE_STEP] = mode_step
            self.state_store.update_state(current_state)

    def set_mode(self, mode):
        # train
        if mode in ALLOWED_MODES:
            self.mode = mode
        else:
            raise ValueError(
                "Invalid mode {}. Valid modes are {}.".format(mode, ",".join(ALLOWED_MODE_NAMES))
            )

        if mode not in self.mode_steps:
            self.mode_steps[mode] = self.init_step

        self._collections_to_save_for_step = None

    def export_collections(self):
        num_workers = self._get_num_workers()
        if num_workers == 1 and self.first_process is False:
            self.logger.warn(
                f"Unsupported Distributed Training Strategy Detected. \
                Sagemaker-Debugger will only write from one process. \
                The process with pid: {os.getpid()} will not be writing any data. \n"
            )
            return
        if self.save_all_workers is False:
            if self.chief_worker != self.worker:
                return
            num_workers = 1  # Override
        self.collection_manager.set_num_workers(num_workers)
        collection_file_name = f"{self.worker}_collections.json"
        self.collection_manager.export(self.out_dir, collection_file_name)

    def _get_reduction_tensor_name(self, tensor_name, reduction_name, abs):
        return get_reduction_tensor_name(tensor_name, reduction_name, abs, remove_colon_index=True)

    def _write_reduction(self, tensor_name, tensor_value, reduction_name, abs, tensor_ref=None):
        reduction_tensor_name = self._get_reduction_tensor_name(tensor_name, reduction_name, abs)
        try:
            tensor_data = self._get_reduction_of_data(
                reduction_name, tensor_value, tensor_name, abs
            )
            self._write_raw_tensor_simple(reduction_tensor_name, tensor_data, tensor_ref=tensor_ref)
        except ValueError as e:
            self.logger.warning(
                f"Could not compute reduction {reduction_name} of {tensor_name} due to {e}"
            )

    def _write_reductions(self, tensor_name, tensor_value, save_collections, tensor_ref=None):
        reductions_saved = set()
        for s_col in save_collections:
            if s_col.name in SCALAR_COLLECTIONS:
                continue
            reduction_config = s_col.reduction_config
            for reduction_list in (reduction_config.reductions, reduction_config.norms):
                for reduction in reduction_list:
                    if (reduction, False) not in reductions_saved:
                        self._write_reduction(
                            tensor_name, tensor_value, reduction, abs=False, tensor_ref=tensor_ref
                        )
                        reductions_saved.add((reduction, False))
            for reduction_list in (reduction_config.abs_reductions, reduction_config.abs_norms):
                for reduction in reduction_list:
                    if (reduction, True) not in reductions_saved:
                        self._write_reduction(
                            tensor_name, tensor_value, reduction, abs=True, tensor_ref=tensor_ref
                        )
                        reductions_saved.add((reduction, True))

    def _write_scalar_summary(self, tensor_name, tensor_value, save_colls):
        """ Maybe write to TensorBoard. """
        tb_writer = self._maybe_get_tb_writer()
        if tb_writer:
            for s_col in save_colls:
                if s_col.name in SCALAR_COLLECTIONS:
                    np_val = self._make_numpy_array(tensor_value)
                    if self.dry_run:
                        return

                    if np_val.squeeze().ndim == 0:
                        self.logger.debug(
                            f"Saving scalar summary {tensor_name} for global step {self.step}"
                        )
                        tb_writer.write_scalar_summary(tensor_name, np_val, self.step)
                    else:
                        self.logger.debug(
                            f"Value of {tensor_name} is not scalar, "
                            f"so scalar summary could not be created"
                        )
                    break

    def _write_histogram_summary(self, tensor_name, tensor_value, save_collections):
        """ Maybe write to TensorBoard. """
        tb_writer = self._maybe_get_tb_writer()
        if tb_writer:
            for s_col in save_collections:
                if s_col.name in NON_HISTOGRAM_COLLECTIONS:
                    continue
                elif s_col.save_histogram is True:
                    np_value = self._make_numpy_array(tensor_value)
                    if self.dry_run or np_value.dtype == np.bool or np_value.nbytes == 0:
                        return

                    hist_name = f"{s_col.name}/{tensor_name}"
                    self.logger.debug(f"Saving {hist_name} for global step {self.step}")
                    tb_writer.write_histogram_summary(
                        tdata=np_value, tname=hist_name, global_step=self.step
                    )
                    break

    def _write_scalars(self):
        """
        This function writes all the scalar values saved in the scalar_cache to file.
        If sm_metric is set to True for certain scalars, then that scalar is written to
        SageMaker as well. By default, loss values are sm_metric.
        """
        if self._is_not_supported():
            # Do not log scalars if smdebug hook is not supported
            # Like when TFDistributionStrategy.UNSUPPORTED
            self.scalar_cache = []
            return
        for scalar_obj in self.scalar_cache:
            scalar_name = scalar_obj.name
            scalar_val = scalar_obj.value
            scalar_mode = scalar_obj.mode
            sm_metric = scalar_obj.sm_metric
            write_tb = scalar_obj.write_tb
            write_event = scalar_obj.write_event
            timestamp = scalar_obj.timestamp
            if self.metrics_writer and sm_metric:
                self.metrics_writer.log_metric(
                    scalar_name + "_" + scalar_mode.name,
                    scalar_val,
                    timestamp=timestamp,
                    iteration_number=self.mode_steps[scalar_mode],
                )
            if write_tb:
                tb_writer = self._maybe_get_tb_writer()
                if tb_writer:
                    tb_writer.write_scalar_summary(
                        scalar_name, scalar_val, self.step, timestamp=timestamp
                    )
            if write_event:
                self._initialize_writers(only_initialize_if_missing=True)
                self._write_raw_tensor_simple(scalar_name, scalar_val, timestamp=timestamp)

        self.scalar_cache = []

    # Fix step number for saving scalar and tensor
    def save_scalar(self, name, value, sm_metric=False, timestamp: float = None):
        """
        Call save_scalar at any point in the training script to log a scalar value,
        such as a metric or any other value.
        :param name: Name of the scalar. A prefix 'scalar/' will be added to it
        :param value: Scalar value
        :param sm_metric: True/False. If set to True, the scalar value will be written to
        SageMaker
        """
        name = CallbackHook.SCALAR_PREFIX + name
        val = self._make_numpy_array(value)
        if val.size != 1:
            raise TypeError(f"{name} has non scalar value of type: {type(value)}")
        scalar_obj = ScalarCache(
            name, val, self.mode, sm_metric, write_tb=True, write_event=True, timestamp=timestamp
        )
        self.scalar_cache.append(scalar_obj)

    def _write_raw_tensor(self, tensor_name, tensor_value, save_collections, tensor_ref=None):
        for s_col in save_collections:
            reduction_config = s_col.reduction_config
            if reduction_config.save_raw_tensor is True:
                self._write_raw_tensor_simple(tensor_name, tensor_value, tensor_ref=tensor_ref)
                break

    def _write_raw_tensor_simple(self, tensor_name, tensor_value, tensor_ref=None, timestamp=None):
        # tensor_ref is used by TF
        # todo: if fp16, check perf of saving as fp16 in proto vs as fp32
        numpy_tensor_value = self._make_numpy_array(tensor_value)
        this_size, this_shape = size_and_shape(numpy_tensor_value)
        if self.dry_run is False and this_size > 0:
            writers = self._get_writers(tensor_name, tensor_ref=tensor_ref)
            for writer in writers:
                writer.write_tensor(
                    tdata=numpy_tensor_value,
                    tname=tensor_name,
                    mode=self.mode,
                    mode_step=self.mode_steps[self.mode],
                    timestamp=timestamp,
                )

    def _save_for_tensor(self, tensor_name, tensor_value, check_before_write=True):
        """
        Identifies if this tensor should be saved for this step
        based on the save configs for the collections it belongs to.
        If this tensor is to be saved, calls write_for_tensor.

        This check can be disabled by passing check_before_write=False.
        Disabling this check is cleaner for TF, as for TF this method is never
        called if tensor should not be saved for this step.
        :param tensor_name: str
        The name of tensor. In TensorFlow's case, this is graph name of tensor
        and will be converted to internal name in write_for_tensor.
        :param tensor_value: dtype is tensor class of corresponding framework
            value of the tensor to be saved
        :param check_before_write: bool
            checks whether to save tensor for this step
        :return:
        """
        save_collections = self._get_collections_with_tensor(tensor_name)
        save_collections_for_tensor = save_collections.intersection(
            self._get_collections_to_save_for_step()
        )
        if check_before_write and bool(save_collections_for_tensor) is False:
            return
        elif not check_before_write:
            # if not checking before write, means we want to write
            # regardless of whether the collection should be written for step
            save_collections_for_tensor = save_collections

        self._write_for_tensor(tensor_name, tensor_value, save_collections_for_tensor)
        for s_col in save_collections_for_tensor:
            if s_col.name in SM_METRIC_COLLECTIONS:
                np_val = self._make_numpy_array(tensor_value)
                # Always log loss to SageMaker
                tensor_val = np.mean(np_val)
                scalar_obj = ScalarCache(
                    tensor_name,
                    tensor_val,
                    self.mode,
                    sm_metric=True,
                    write_tb=False,
                    write_event=False,
                )
                self.scalar_cache.append(scalar_obj)

    def _log_save(self, tensor_name, save_collections):
        coll_str = ", ".join([x.name for x in save_collections])
        many_colls = len(save_collections) > 1
        if self.mode != ModeKeys.GLOBAL:
            step_str = f"for step {self.mode_steps[self.mode]} of mode {self.mode.name}"
        else:
            step_str = f"for step: {self.step}"
        base_str = f"Saving {tensor_name} from {'collections' if many_colls else 'collection'}"
        self.logger.debug(f"{base_str} {coll_str} {step_str}")

    def _write_for_tensor(self, tensor_name, tensor_value, save_collections, tensor_ref=None):
        """
        Write all data that we might want to for this tensor
        :param tensor_name: name of tensor
        :param tensor_value: value (could be in framework tensor dtype)
        :param save_collections: list of collections which are being saved for this step
        """
        self._log_save(tensor_name, save_collections)
        # write reductions defined for collections this tensor may be part of
        self._write_reductions(tensor_name, tensor_value, save_collections, tensor_ref=tensor_ref)

        # write histogram for this tensor if any collection this tensor
        # is part of has save_histogram as True
        self._write_histogram_summary(tensor_name, tensor_value, save_collections)

        # write raw tensor if save_raw_tensor in reduction config is True
        self._write_raw_tensor(tensor_name, tensor_value, save_collections, tensor_ref=tensor_ref)

        # writes scalar summary if this value is a scalar (or 1x1 array)
        self._write_scalar_summary(tensor_name, tensor_value, save_collections)

    @staticmethod
    @abstractmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        """
        Returns the reduction of given tensor
        :param reduction_name: str
            type of reduction
        :param tensor_value: tensor_data_type
            reduction to be performed on this original tensor value
        :param tensor_name: str
            name of original tensor
        :param abs: bool
            whether to take absolute value of tensor before performing reduction
        :return:
        """

    @staticmethod
    @abstractmethod
    def _make_numpy_array(tensor_value):
        """
        Convert the tensor value into a numpy array
        :param tensor_value: mx.nd.NDArray, torch.Tensor, etc
        :return: numpy ndarray
        """

    def get_collection(self, name, create=True):
        return self.collection_manager.get(name, create=create)

    def get_collections(self):
        return self.collection_manager.get_collections()

    def add_collection(self, collection):
        if not isinstance(collection, Collection):
            raise TypeError(
                f"collection must be an instance of Collection class. "
                f"value of type {collection.__class__} is not supported"
            )
        self.collection_manager.add(collection)


class CallbackHook(BaseHook):
    __metaclass__ = ABCMeta
    INVALID_TAG_CHARACTERS = _re.compile(r"[^-/\w\.]")
    INPUT_TENSOR_SUFFIX = "_input_"
    OUTPUT_TENSOR_SUFFIX = "_output_"
    GRADIENT_PREFIX = "gradient/"
    SCALAR_PREFIX = "scalar/"

    def __init__(
        self,
        collection_manager: CollectionManager,
        default_include_collections: List[str],
        data_type_name: Optional[str] = None,
        out_dir: Optional[str] = None,
        export_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
        dry_run: bool = False,
        reduction_config: Optional[ReductionConfig] = None,
        save_config: Optional[SaveConfig] = None,
        include_regex: Optional[List[str]] = None,
        include_collections: Optional[List[str]] = None,
        save_all: bool = False,
        include_workers: str = "one",
    ):
        super().__init__(
            collection_manager=collection_manager,
            default_include_collections=default_include_collections,
            init_step=-1,
            out_dir=out_dir,
            export_tensorboard=export_tensorboard,
            tensorboard_dir=tensorboard_dir,
            dry_run=dry_run,
            reduction_config=reduction_config,
            save_config=save_config,
            include_regex=include_regex,
            include_collections=include_collections,
            save_all=save_all,
            include_workers=include_workers,
        )
        self.exported_collections = False
        self.data_type_name = data_type_name

    def _cleanup(self):
        if not self.exported_collections:
            self.export_collections()
            self.exported_collections = True
        super()._cleanup()

    def _write(self, module_name, var, suffix, idx):
        if self.data_type_name is None:
            raise RuntimeError("This method can not be called when data_type is None")

        if var.__class__.__name__ == self.data_type_name:
            self._save_for_tensor(module_name + suffix + str(idx), var)
            return idx + 1
        elif isinstance(var, tuple) or isinstance(var, list):
            for val in var:
                idx = self._write(module_name, val, suffix, idx)
        else:
            logger.warning(
                f"var is not {self.data_type_name} or list or tuple "
                f"of {self.data_type_name}s, "
                f"module_name:{module_name} {var.__class__.__name__}"
            )
        return idx

    def _write_inputs(self, name, inputs):
        tensor_name = name + CallbackHook.INPUT_TENSOR_SUFFIX
        idx = self.written_tensor_name_for_step.get(tensor_name, 0)
        self.written_tensor_name_for_step[tensor_name] = self._write(
            name, inputs, CallbackHook.INPUT_TENSOR_SUFFIX, idx=idx
        )

    def _write_outputs(self, name, outputs):
        tensor_name = name + CallbackHook.OUTPUT_TENSOR_SUFFIX
        idx = self.written_tensor_name_for_step.get(tensor_name, 0)
        self.written_tensor_name_for_step[tensor_name] = self._write(
            name, outputs, CallbackHook.OUTPUT_TENSOR_SUFFIX, idx=idx
        )

    @abstractmethod
    def _export_model(self):
        pass
