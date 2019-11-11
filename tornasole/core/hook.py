import atexit
from abc import ABCMeta, abstractmethod
import re as _re
import os
import numpy as np
from typing import Optional, List, Union, Tuple, Dict, Set

from tornasole.core.utils import match_inc, size_and_shape
from tornasole.core.reduction_config import ReductionConfig
from tornasole.core.collection import (
    CollectionKeys,
    SUMMARIES_COLLECTIONS,
    NON_HISTOGRAM_COLLECTIONS,
)
from tornasole.core.collection_manager import CollectionManager
from tornasole.core.save_config import SaveConfig, SaveConfigMode
from tornasole.core.access_layer import training_has_ended
from tornasole.core.hook_utils import verify_and_get_out_dir, get_tensorboard_dir
from tornasole.core.sagemaker_utils import is_sagemaker_job
from tornasole.core.modes import ModeKeys, ALLOWED_MODES
from tornasole.core.utils import flatten, get_tb_worker
from tornasole.core.logger import get_logger
from tornasole.core.reductions import get_reduction_tensor_name
from tornasole.core.writer import FileWriter
from tornasole.core.state_store import StateStore
from tornasole.core.config_constants import (
    TRAINING_RUN,
    LATEST_GLOBAL_STEP_SAVED,
    LATEST_GLOBAL_STEP_SEEN,
    LATEST_MODE_STEP,
)


logger = get_logger()


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
    ):
        """
        A class used to represent the hook which gets attached to the
        training process. This takes the form appropriate for the framework
        such as tf.train.SessionRunHook for TF, Callback for keras...

        ...

        Attributes
        ----------
        out_dir : str
            represents a path into which tornasole outputs will be written to
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
        """
        self.out_dir = verify_and_get_out_dir(out_dir)
        self.tensorboard_dir = get_tensorboard_dir(
            export_tensorboard=export_tensorboard,
            tensorboard_dir=tensorboard_dir,
            out_dir=self.out_dir,
        )

        self.dry_run = dry_run
        self.worker = None
        if include_collections is None:
            include_collections = default_include_collections
        self.default_include_collections = default_include_collections
        self.include_collections = flatten(include_collections)

        self.save_all = save_all
        self.save_config = SaveConfig.parse(save_config)
        if reduction_config is None:
            reduction_config = ReductionConfig(save_raw_tensor=True)
        self.reduction_config = reduction_config
        self.include_regex = include_regex
        self.collection_manager = collection_manager
        self.collection_manager.set_num_workers(self.get_num_workers())
        self.init_step = init_step

        self.logger = logger

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
        # Maps ModeKeys to FileWriter objects
        self.tb_writers = {}
        self.logger.info("Saving to {}".format(self.out_dir))
        atexit.register(self._cleanup)

        # Check if there is any last saved tornasole state. Initialize the hook based last saved state.
        self.training_run = 0
        self._initialize_to_last_saved_state()

    def _initialize_to_last_saved_state(self):
        self.state_store = StateStore()
        last_tornasole_state = self.state_store.get_last_saved_tornasole_state()
        if last_tornasole_state is not None:
            self.last_saved_step = last_tornasole_state[LATEST_GLOBAL_STEP_SAVED]
            self.init_step = last_tornasole_state[LATEST_GLOBAL_STEP_SEEN]
            self.training_run = 1 + last_tornasole_state[TRAINING_RUN]
            for (mode, step) in last_tornasole_state[LATEST_MODE_STEP].items():
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

    @abstractmethod
    def get_worker_name(self):
        pass

    @abstractmethod
    def get_num_workers(self):
        pass

    #### Save Manager methods ####

    def _should_collection_be_saved(self, coll_name: str) -> bool:
        return coll_name in self.include_collections

    def _assert_prep(self):
        assert self.prepared_collections, "Collections have not been prepared yet"

    def _get_all_collections_to_save(self) -> Set["Collection"]:
        self._assert_prep()
        return self._collections_to_save

    def _get_collections_to_save_for_step(self) -> Set["Collection"]:
        if self._collections_to_save_for_step is None:
            self._assert_prep()
            s = set()
            for coll in self._collections_to_save:
                if coll.name == CollectionKeys.GRADIENTS and self.mode in [
                    ModeKeys.EVAL,
                    ModeKeys.PREDICT,
                ]:
                    continue
                if coll.save_config.should_save_step(self.mode, self.mode_steps[self.mode]):
                    s.add(coll)
            self._collections_to_save_for_step = s
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

    def _should_save_tensor_for_step(self, tensorname) -> bool:
        """Returns whether tensorname should be saved for this mode, mode_step
        as a bool
        """
        for coll in self._get_collections_with_tensor(tensorname):
            if coll in self._get_collections_to_save_for_step():
                return True
        return False

    def _prepare_collections(self):
        """Populate collections_to_save and ensure every collection has
        a save_config and reduction_config."""
        for c_name, c in self.collection_manager.get_collections().items():
            if self._should_collection_be_saved(c_name) and c not in self._collections_to_save:
                self._collections_to_save.add(c)

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

            if c_name in SUMMARIES_COLLECTIONS:
                c.reduction_config = ReductionConfig(save_raw_tensor=True)
            elif c.reduction_config is None:
                c.reduction_config = self.reduction_config

        self.prepared_collections = True

    #### End of Save Manager methods ####

    def _close_writer(self) -> None:
        if self.dry_run:
            return

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _close_writers(self) -> None:
        if self.dry_run:
            return

        self._close_writer()
        to_delete_writers = []

        # Delete all the tb writers
        for mode, writer in self.tb_writers.items():
            if writer is not None:
                writer.flush()
                writer.close()
                to_delete_writers.append(mode)
        for mode in to_delete_writers:
            del self.tb_writers[mode]

    def _initialize_writer(self) -> None:
        if self.dry_run:
            return
        self.writer = FileWriter(trial_dir=self.out_dir, step=self.step, worker=self.worker)

    def get_writers(self, tensor_name) -> List[FileWriter]:
        """
        :param tensor_name:
        :return: List[FileWriter]
        """
        return [self.writer]

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
        training_has_ended(self.out_dir)

    def _increment_step(self):
        # Update the last_tornasole_state to the last step number that was saved or seen
        self._write_tornasole_state()

        self.step += 1
        self.mode_steps[self.mode] += 1
        self._collections_to_save_for_step = None

    def _write_tornasole_state(self):
        if self.state_store.is_checkpoint_updated():
            current_tornasole_state = dict()
            current_tornasole_state[TRAINING_RUN] = self.training_run
            current_tornasole_state[LATEST_GLOBAL_STEP_SAVED] = self.last_saved_step
            current_tornasole_state[LATEST_GLOBAL_STEP_SEEN] = self.step
            mode_step = dict()
            for (mode, step) in self.mode_steps.items():
                mode_step[mode.name] = step
            current_tornasole_state[LATEST_MODE_STEP] = mode_step
            self.state_store.update_tornasole_state(current_tornasole_state)

    def set_mode(self, mode):
        # train
        if mode in ALLOWED_MODES:
            self.mode = mode
        else:
            raise ValueError(
                "Invalid mode {}. Valid modes are {}.".format(mode, ",".join(ALLOWED_MODES))
            )

        if mode not in self.mode_steps:
            self.mode_steps[mode] = self.init_step

        self._collections_to_save_for_step = None

    def export_collections(self):
        self.collection_manager.set_num_workers(self.get_num_workers())
        collection_file_name = f"{self.worker}_collections.json"
        self.collection_manager.export(self.out_dir, collection_file_name)

    def _write_reduction(self, tensor_name, tensor_value, reduction_name, abs):
        reduction_tensor_name = get_reduction_tensor_name(tensor_name, reduction_name, abs)
        tensor_data = self._get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs)
        self._write_raw_tensor_simple(reduction_tensor_name, tensor_data)

    def _write_reductions(self, tensor_name, tensor_value, save_collections):
        reductions_saved = set()
        for s_col in save_collections:
            reduction_config = s_col.reduction_config
            for reduction_list in (reduction_config.reductions, reduction_config.norms):
                for reduction in reduction_list:
                    if (reduction, abs) not in reductions_saved:
                        self._write_reduction(tensor_name, tensor_value, reduction, abs=False)
                        reductions_saved.add((reduction, False))
            for reduction_list in (reduction_config.abs_reductions, reduction_config.abs_norms):
                for reduction in reduction_list:
                    if (reduction, abs) not in reductions_saved:
                        self._write_reduction(tensor_name, tensor_value, reduction, abs=True)
                        reductions_saved.add((reduction, True))

    def _write_scalar_summary(self, tensor_name, tensor_value, save_colls):
        """ Maybe write to TensorBoard. """
        tb_writer = self._maybe_get_tb_writer()
        if tb_writer:
            for s_col in save_colls:
                if s_col.name in [CollectionKeys.LOSSES, CollectionKeys.SCALARS]:
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
                if s_col.name in SUMMARIES_COLLECTIONS:
                    continue
                elif s_col.save_histogram is True:
                    np_value = self._make_numpy_array(tensor_value)
                    if self.dry_run or np_value.dtype == np.bool or np_value.nbytes == 0:
                        return

                    hist_name = f"histograms/{s_col.name}/{tensor_name}"
                    self.logger.debug(f"Saving {hist_name} for global step {self.step}")
                    tb_writer.write_histogram_summary(
                        tdata=np_value, tname=hist_name, global_step=self.step
                    )
                    break

    # Fix step number for saving scalar and tensor
    # def save_scalar(self, name, value):
    #     get_collection(CollectionKeys.SCALARS).add_tensor_name(name)
    #     if self.writer is None:
    #         self._init_writer()
    #     val = make_numpy_array(value)
    #     if val.size != 1:
    #         raise TypeError(
    #             f'{name} has non scalar value of type: {type(value)}')
    #     self._save_scalar_summary(name, val)
    #     logger.debug(f'Saving scalar {name} {val} for step {self.step} {self.mode} {self.mode_steps[self.mode]}')
    #     self._save_raw_tensor(name, val)

    # def save_tensor(self, name, value):
    #     # todo: support to add these tensors to any collection.
    #     #  complication here is that we need to export the file again
    #     # todo: what happens if name is conflicting
    #     if self.writer is None:
    #         self._init_writer()
    #     self._save_raw_tensor(name, value)

    def _write_raw_tensor(self, tensor_name, tensor_value, save_collections):
        for s_col in save_collections:
            reduction_config = s_col.reduction_config
            if reduction_config.save_raw_tensor is True:
                self._write_raw_tensor_simple(tensor_name, tensor_value)
                break

    def _write_raw_tensor_simple(self, tensor_name, tensor_value):
        # todo: if fp16, check perf of saving as fp16 in proto vs as fp32
        numpy_tensor_value = self._make_numpy_array(tensor_value)
        this_size, this_shape = size_and_shape(numpy_tensor_value)
        if self.dry_run is False and this_size > 0:
            writers = self.get_writers(tensor_name)
            for writer in writers:
                writer.write_tensor(
                    tdata=numpy_tensor_value,
                    tname=tensor_name,
                    mode=self.mode,
                    mode_step=self.mode_steps[self.mode],
                )

    def _save_for_tensor(self, tensor_name, tensor_value, check_before_write=True):
        # for TF, the tensor_name coming in will the name of object in graph
        # it is converted to tornasole_name in write_for_tensor
        if (
            check_before_write
            and self._should_save_tensor_for_step(tensorname=tensor_name) is False
        ):
            return

        save_collections = self._get_collections_with_tensor(tensor_name)
        save_collections_for_tensor = save_collections.intersection(
            self._get_collections_to_save_for_step()
        )
        self._write_for_tensor(tensor_name, tensor_value, save_collections_for_tensor)

    def _write_for_tensor(self, tensor_name, tensor_value, save_collections):
        """
        Write all data that we might want to for this tensor
        :param tensor_name: name of tensor
        :param tensor_value: value (could be in framework tensor dtype)
        :param save_collections: list of collections which are being saved for this step
        """
        self.logger.debug(f"Saving {tensor_name} for global step {self.step}")
        # write reductions defined for collections this tensor may be part of
        self._write_reductions(tensor_name, tensor_value, save_collections)

        # write histogram for this tensor if any collection this tensor
        # is part of has save_histogram as True
        self._write_histogram_summary(tensor_name, tensor_value, save_collections)

        # write raw tensor if save_raw_tensor in reduction config is True
        self._write_raw_tensor(tensor_name, tensor_value, save_collections)

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
        pass

    @staticmethod
    @abstractmethod
    def _make_numpy_array(tensor_value):
        """
        Convert the tensor value into a numpy array
        :param tensor_value: mx.nd.NDArray, torch.Tensor, etc
        :return: numpy ndarray
        """
        pass


class CallbackHook(BaseHook):
    __metaclass__ = ABCMeta
    INVALID_TAG_CHARACTERS = _re.compile(r"[^-/\w\.]")
    INPUT_TENSOR_SUFFIX = "_input_"
    OUTPUT_TENSOR_SUFFIX = "_output_"
    GRADIENT_PREFIX = "gradient/"

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

    def _write_inputs(self, name, inputs):
        self._write(name, inputs, CallbackHook.INPUT_TENSOR_SUFFIX, idx=0)

    def _write_outputs(self, name, outputs):
        self._write(name, outputs, CallbackHook.OUTPUT_TENSOR_SUFFIX, idx=0)

    @abstractmethod
    def _export_model(self):
        pass
