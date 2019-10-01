import atexit
import socket
from abc import ABCMeta, abstractmethod
import re as _re
import os
from typing import Optional, List, Union, Tuple, Dict, Set

from tornasole.core.utils import match_inc
from tornasole.core.reduction_config import ReductionConfig
from tornasole.core.collection_manager import CollectionManager, COLLECTIONS_FILE_NAME
from tornasole.core.collection import CollectionKeys
from tornasole.core.save_config import SaveConfig, SaveConfigMode
from tornasole.core.access_layer import training_has_ended
from tornasole.core.hook_utils import verify_and_get_out_dir
from tornasole.core.modes import ModeKeys, ALLOWED_MODES
from tornasole.core.utils import flatten
from tornasole.core.logger import get_logger
from tornasole.core.json_config import TORNASOLE_CONFIG_DEFAULT_WORKER_NAME
from tornasole.core.reductions import get_reduction_tensor_name
from tornasole.core.writer import FileWriter

logger = get_logger()


class BaseHook:
    __metaclass__ = ABCMeta

    def __init__(self,
                 collection_manager: CollectionManager,
                 default_include_collections: List[str],
                 init_step: int = 0,
                 out_dir: Optional[str] = None,
                 dry_run: bool = False,
                 worker: str = TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
                 reduction_config: Optional[ReductionConfig] = None,
                 save_config: Optional[Union[SaveConfig, Dict[ModeKeys, SaveConfigMode]]] = None,
                 include_regex: Optional[List[str]] = None,
                 include_collections: Optional[List[str]] = None,
                 save_all: bool = False):
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
        worker: string
            name of worker in a multi process training job
            outputs and tensors are organized by this name during retrieval.

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

        self.dry_run = dry_run
        self.worker = worker if worker is not None else socket.gethostname()
        if include_collections is None:
            include_collections = default_include_collections
        self.default_include_collections = default_include_collections
        self.include_collections = flatten(include_collections)

        self.save_all = save_all
        self.save_config = SaveConfig.parse(save_config)
        self.reduction_config = reduction_config
        self.include_regex = include_regex
        self.collection_manager = collection_manager
        self.init_step = init_step

        self.logger = logger

        if include_regex is not None:
            collection_manager.get(CollectionKeys.DEFAULT).include(include_regex)
            if CollectionKeys.DEFAULT not in self.include_collections:
                self.include_collections.append(CollectionKeys.DEFAULT)

        self.save_all = save_all
        if self.save_all:
            collection_manager.get(CollectionKeys.ALL).include('.*')
            if CollectionKeys.ALL not in self.include_collections:
                self.include_collections.append(CollectionKeys.ALL)

        if CollectionKeys.DEFAULT not in self.include_collections and \
                collection_manager.get(CollectionKeys.DEFAULT).get_include_regex():
            self.logger.warn('The `default` collection was not passed to '
                             'include_collections. So it is not being saved')

        self.prepared_collections = False
        self._collections_to_save = set()
        # todo clear cache for old steps
        self.save_states_cache = {}
        self.tensor_to_collections = {}
        self.step = init_step
        self.mode = ModeKeys.GLOBAL
        self.mode_steps = {ModeKeys.GLOBAL: init_step}
        self.writer = None
        self.logger.info('Saving to {}'.format(self.out_dir))
        atexit.register(self._cleanup)

    #### Save Manager methods ####

    def _should_collection_be_saved(self, coll_name: str) -> bool:
        return coll_name in self.include_collections

    def _assert_prep(self):
        assert self.prepared_collections, \
            "Collections have not been prepared yet"

    def _get_all_collections_to_save(self) -> Set['Collection']:
        self._assert_prep()
        return self._collections_to_save

    def _get_collections_to_save_for_step(self, mode, step) -> Set['Collection']:
        """Mark the proper collections to be saved, return a set of those."""
        self._assert_prep()
        if (mode, step) not in self.save_states_cache:
            coll_to_save_for_step = set()
            for coll in self._collections_to_save:
                if coll.get_save_config().should_save_step(mode, step):
                    coll_to_save_for_step.add(coll)
            self.save_states_cache[(mode, step)] = coll_to_save_for_step
        return self.save_states_cache[(mode, step)]

    def _get_collections_with_tensor(self, tensor_name) -> Set['Collection']:
        self._assert_prep()
        # for tf this will be prepopulated because of prepare_tensors
        if not tensor_name in self.tensor_to_collections:
            # for mxnet it is computed and then cached
            matched_colls = set()
            for coll in self._collections_to_save:
                if tensor_name in coll.tensor_names:
                    # if being matched as reduction,
                    # it must be in reduction_tensor_name, not with regex
                    matched_colls.add(coll)
                elif match_inc(tensor_name, coll.get_include_regex()):
                    coll.add_tensor_name(tensor_name)
                    matched_colls.add(coll)
            self.tensor_to_collections[tensor_name] = matched_colls
        return self.tensor_to_collections[tensor_name]

    def _should_save_tensor_for_step(self, tensorname, mode, step) -> bool:
        """Returns whether tensorname should be saved for this mode, mode_step
        as a bool
        """
        colls_to_save = self._get_collections_to_save_for_step(mode, step)
        for coll in self._get_collections_with_tensor(tensorname):
            if coll in colls_to_save:
                return True
        return False

    def _prepare_collections(self):
        """Populate collections_to_save and ensure every collection has
        a save_config and reduction_config."""
        for c_name, c in self.collection_manager.get_collections().items():
            if self._should_collection_be_saved(c_name) \
                    and c not in self._collections_to_save:
                self._collections_to_save.add(c)

        # Populate configs_for_collections and reduction_config
        for c_name, c in self.collection_manager.get_collections().items():
            if c.save_config is None:
                # Set to the default if None
                c.save_config = self.save_config
            elif isinstance(c.save_config, SaveConfig):
                # Otherwise, set missing modes to the defaults
                c.save_config.merge_default_save_config(self.save_config)
            else:
                raise TypeError(
                    f"save_config={c.save_config} must be None or SaveConfig")

            if c.get_reduction_config() is None and self.reduction_config is not None:
                c.set_reduction_config(self.reduction_config)
        self.prepared_collections = True

    #### End of Save Manager methods ####

    def _flush_and_close_writer(self) -> None:
        if self.dry_run:
            return
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _initialize_writer(self) -> None:
        if self.dry_run:
            return
        self.writer = FileWriter(
                trial_dir=self.out_dir, step=self.step, worker=self.worker)

    def _cleanup(self):
        self._flush_and_close_writer()
        training_has_ended(self.out_dir)

    def _increment_step(self):
        self.step += 1
        self.mode_steps[self.mode] += 1

    def set_mode(self, mode):
        # train
        if mode in ALLOWED_MODES:
            self.mode = mode
        else:
            raise ValueError('Invalid mode {}. Valid modes are {}.'
                             .format(mode, ','.join(ALLOWED_MODES)))

        if mode not in self.mode_steps:
            self.mode_steps[mode] = self.init_step

    def export_collections(self):
        self.collection_manager.export(os.path.join(
                self.out_dir, COLLECTIONS_FILE_NAME))



class CallbackHook(BaseHook):
    __metaclass__ = ABCMeta
    INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')
    INPUT_TENSOR_SUFFIX = '_input_'
    OUTPUT_TENSOR_SUFFIX = '_output'
    GRADIENT_PREFIX = 'gradient/'

    def __init__(self,
                 collection_manager: CollectionManager,
                 default_include_collections: List[str],
                 data_type_name: Optional[str] = None,
                 out_dir: Optional[str] = None,
                 dry_run: bool = False,
                 worker: str = TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
                 reduction_config: Optional[ReductionConfig] = None,
                 save_config: Optional[SaveConfig] = None,
                 include_regex: Optional[List[str]] = None,
                 include_collections: Optional[List[str]] = None,
                 save_all: bool = False):
        super().__init__(collection_manager=collection_manager,
                         default_include_collections=default_include_collections,
                         init_step=-1,
                         out_dir=out_dir,
                         dry_run=dry_run,
                         worker=worker,
                         reduction_config=reduction_config,
                         save_config=save_config,
                         include_regex=include_regex,
                         include_collections=include_collections,
                         save_all=save_all)
        self.last_saved_step = None
        self.exported_collections = False
        self.data_type_name = data_type_name
        # collections that need to be saved in a particular step.
        self.collections_in_this_step = None

    def _cleanup(self):
        if not self.exported_collections:
            self.export_collections()
            self.exported_collections = True
        super()._cleanup()

    def _process_step(self) -> Set['Collection']:
        # returns set of collections which need to be saved for step
        self.collections_in_this_step = self._get_collections_to_save_for_step(
            self.mode, self.mode_steps[self.mode])
        return self.collections_in_this_step

    def _write(self, module_name, var, suffix, idx):
        if self.data_type_name is None:
            raise RuntimeError(
                    "This method can not be called when data_type is None")

        if var.__class__.__name__ == self.data_type_name:
            self._write_tensor(module_name + suffix + str(idx), var)
            return idx + 1
        elif isinstance(var, tuple) or isinstance(var, list):
            for val in var:
                idx = self._write(module_name, val, suffix, idx)
        else:
            logger.warning(
                    f"var is not {self.data_type_name} or list or tuple "
                    f"of {self.data_type_name}s, "
                    f"module_name:{module_name} {var.__class__.__name__}")

    def _write_inputs(self, name, inputs):
        self._write(name, inputs, CallbackHook.INPUT_TENSOR_SUFFIX, idx=0)

    def _write_outputs(self, name, outputs):
        self._write(name, outputs, CallbackHook.OUTPUT_TENSOR_SUFFIX, idx=0)

    def _write_reduction(self, tensor_name, tensor_value, reduction_name, abs):
        reduction_tensor_name = get_reduction_tensor_name(
                tensor_name, reduction_name, abs)
        tensor_data = self._get_reduction_of_data(
                reduction_name, tensor_value, tensor_name, abs)
        tensor_value_np = self._make_numpy_array(tensor_data)
        self.writer.write_tensor(tdata=tensor_value_np,
                                 tname=reduction_tensor_name,
                                 mode=self.mode,
                                 mode_step=self.mode_steps[self.mode])

    def _write_reductions(self, tensor_name, tensor_value, reduction_config):
        for reduction_list in (reduction_config.reductions,
                         reduction_config.norms):
            for reduction in reduction_list:
                self._write_reduction(
                        tensor_name, tensor_value, reduction, abs=False)
        for reduction_list in (reduction_config.abs_reductions,
                         reduction_config.abs_norms):
            for reduction in reduction_list:
                self._write_reduction(
                        tensor_name, tensor_value, reduction, abs=True)

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

    def _write_tensor(self, tensor_name, tensor_value):
        if self.dry_run or \
                self._should_save_tensor_for_step(
                        tensorname=tensor_name, mode=self.mode,
                        step=self.mode_steps[self.mode]) is False:
            return

        save_collections = self._get_collections_with_tensor(tensor_name)
        for save_collection in save_collections:
            if save_collection in self.collections_in_this_step:
                reduction_config = save_collection.get_reduction_config()
                if reduction_config is not None:
                    self._write_reductions(
                            tensor_name, tensor_value, reduction_config)
                else:
                    tensor_value = self._make_numpy_array(tensor_value)
                    self.writer.write_tensor(
                            tdata=tensor_value,
                            tname=tensor_name,
                            mode=self.mode,
                            mode_step=self.mode_steps[self.mode])


    @staticmethod
    def clean_tag(name):
        if name is not None:
            new_name = CallbackHook.INVALID_TAG_CHARACTERS.sub('_', name)
            new_name = new_name.lstrip('/')  # Remove leading slashes
            if new_name != name:
                logger.warning(
                    'Summary name %s is illegal; using %s instead.', name,
                    new_name)
                name = new_name
        return name
