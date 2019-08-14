import mxnet as mx
from tornasole.core.writer import FileWriter
from tornasole.core.save_config import SaveConfig
from tornasole.core.save_manager import SaveManager
from tornasole.core.modes import ModeKeys, ALLOWED_MODES
from tornasole.core.utils import check_dir_exists, get_logger, flatten, is_s3, get_reduction_tensor_name
from tornasole.core.access_layer.utils import training_has_ended
from .mxnet_collection import get_collection_manager, get_collection
from .util import get_aggregated_data, make_numpy_array
import re as _re
import logging
import os


logger = get_logger()
import atexit

INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')
COLLECTION_FILE_NAME = 'collections.ts'
DEFAULT_WORKER_NAME = 'worker0'
INPUT_TENSOR_SUFFIX = '_input_'
OUTPUT_TENSOR_SUFFIX = '_output'
GRADIENT_PREFIX = 'gradient/'


def default_save_config():
    return SaveConfig()


class TornasoleHook:
    def __init__(self,
                 out_dir,
                 dry_run=False,
                 worker=DEFAULT_WORKER_NAME,
                 reduction_config=None,
                 save_config=default_save_config(),
                 include_regex=None,
                 include_collections=['weights', 'bias','gradients', 'default'],
                 save_all=False):
        if not is_s3(out_dir)[0]:
            out_dir = os.path.expanduser(out_dir)
        check_dir_exists(out_dir)
        self.out_dir = out_dir
        self.out_base_dir = os.path.dirname(out_dir)
        self.run_id = os.path.basename(out_dir)
        self.include_collections = include_collections

        self.dry_run = dry_run
        self.worker = worker

        self.mode = ModeKeys.GLOBAL
        self.mode_steps = {ModeKeys.GLOBAL: -1}
        self.local_reductions = []
        self.step = -1
        self.is_recursive = False
        self.export_only_once = True
        self.last_saved_step = -1
        self.writer = None
        self.export_collections = True
        self._initialize_collectors(save_all, include_regex)

        atexit.register(self.cleanup)
        self.last_block = None
        # dictionary of collections that need to be saved in a particular step.
        self.collections_in_this_step = None

        self.save_manager = SaveManager(collection_manager=get_collection_manager(),
                                        include_collections_names=self.include_collections,
                                        default_save_config=save_config,
                                        default_reduction_config=reduction_config)
        self.prepared_save_manager = False
        logger.info('Saving to {}'.format(self.out_dir))

    def _initialize_collectors(self, save_all, include_regex):
        # If user has provided any include_regex, add them to a default collection.
        if include_regex is not None:
            get_collection('default').include(include_regex)
            if 'default' not in self.include_collections:
                self.include_collections.append('default')
        # If save all is set, create a collector that can save all the tensors
        if save_all :
            get_collection('all').include([".*"])
            self.include_collections.append('all')

    def set_mode(self, mode):
        if mode in ALLOWED_MODES:
            self.mode = mode
        else:
            raise ValueError('Invalid mode {}. Valid modes are {}.'
                             .format(mode, ','.join(ALLOWED_MODES)))
        
        if mode not in self.mode_steps:
            self.mode_steps[mode] = -1

    def cleanup(self):
        if logger is not None:
            logger.debug("Cleanup")
        if self.last_saved_step != -1:
            get_collection_manager().export_manager(os.path.join(self.out_dir, COLLECTION_FILE_NAME))
            self.export_only_once = False
        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None:
            if self.last_block is not None:
                params = self.last_block.collect_params().values()
                for param in params:
                    self.log_param(param)
            self.writer.flush()
            self.writer.close()
        training_has_ended(self.out_dir)

    # Check whether we should log this tensor
    def _check_tensor_to_be_logged(self, name):
        ss = self.save_manager.should_save_tensor(tensorname=name, mode=self.mode,
                                                  step=self.mode_steps[self.mode])
        return ss['step']

    def _process_step(self):
        # returns dictionary of dictionaries: coll_name -> {step: True/False, when_nan: True/False}
        # there will be no entry in dictionary for collections where both step and when_nan are False
        # This dictionary is stored in self.collections_in_this_step so that we do not need to call this
        # function in every forward_hook (recursive) invocation for a given step.
        self.collections_in_this_step = self.save_manager.collections_to_save(self.mode, self.mode_steps[self.mode])
        return self.collections_in_this_step

    # This hook is invoked by trainer prior to running the forward pass.
    def forward_pre_hook(self, block, input):
        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None:
            params = block.collect_params().values()
            for param in params:
                self.log_param(param)
            self.writer.flush()
            self.writer.close()
            self.writer = None

        if not self.prepared_save_manager:
            # at this point we need all collections to be ready
            # this may not be the case at creation of hook
            # as user's code after hook might add collections
            self.save_manager.prepare()
            self.prepared_save_manager = True

        self.mode_steps[self.mode] += 1
        self.step += 1
        logger.debug("Setting the global step to be {0}".format(self.step))

        # Reset the collections to be saved in this step to be None.
        self.collections_in_this_step = None
        if self._process_step():
            self.writer = FileWriter(logdir=self.out_base_dir,
                                     trial=self.run_id,
                                     step=self.step,
                                     worker=self.worker)

        if self.last_saved_step != -1 and self.export_only_once:
            get_collection_manager().export_manager(os.path.join(self.out_dir, COLLECTION_FILE_NAME))
            self.export_only_once = False
        self.last_block = block

    # This hook is invoked by trainer after running the forward pass.
    def forward_hook(self, block, input, output):
        if not self.collections_in_this_step:
            logging.debug("Skipping the global step {0}".format(self.step))
            return

        block_name = block.name
        logger.debug("Processing the global step {0} for block {1}".format(self.step, block_name))

        # Output input tensor
        self.log_inputs_to_block(block_name, input)

        # Output output tensors
        self.log_outputs_of_block(block_name, output)
        self.last_saved_step = self.step
 
    def _log_ndarray_from_col(self, block_name, var, tensor_suffix, idx):
       if var.__class__.__name__ is "NDArray":
           self.log_tensor(tensor_name=block_name + tensor_suffix + str(idx), tensor_value=var)
           return idx+1
       elif isinstance(var, tuple) or isinstance(var, list):
           for val in var:
               idx = self._log_ndarray_from_col(block_name, val, tensor_suffix, idx)
       else:
           logger.warning("output is not ndarray or list of ndarrays, bname:{} output_class:{}".format(block_name,
var.__class__.__name__))
       return idx

    def log_inputs_to_block(self, block_name, input):
        idx = 0
        self._log_ndarray_from_col(block_name, input, INPUT_TENSOR_SUFFIX, idx)
 
    def log_outputs_of_block(self, block_name, output):
        idx = 0
        self._log_ndarray_from_col(block_name, output, OUTPUT_TENSOR_SUFFIX, idx)

    def log_param(self, param):
        self.log_tensor(tensor_name=param.name, tensor_value=param.data(param.list_ctx()[0]))
        # If Gradient for this param is available
        if param.grad_req != 'null':
            self.log_tensor(tensor_name=GRADIENT_PREFIX + param.name,
                            tensor_value=param.grad(param.list_ctx()[0]))

    def log_tensor(self, tensor_name, tensor_value):
        if self.dry_run or not self._check_tensor_to_be_logged(tensor_name):
            return

        # Get the collection to which this tensor belongs
        save_colls = self.save_manager.from_collections(tensor_name)
        for s_col in save_colls:
            if s_col.name in self.collections_in_this_step.keys():
                reduce_config = s_col.get_reduction_config()
                if reduce_config:
                    abs = False
                    for reduction in reduce_config.reductions + reduce_config.abs_reductions + reduce_config.norms + \
                                     reduce_config.abs_norms:
                        if reduction in reduce_config.abs_reductions or reduction in reduce_config.abs_norms:
                            abs = True
                        reduction_tensor_name = get_reduction_tensor_name(tensor_name, reduction, abs)
                        tensor_data = get_aggregated_data(reduction, tensor_value, tensor_name, abs)
                        tensor_value_np = make_numpy_array(tensor_data)
                        self.writer.write_tensor(tdata=tensor_value_np, tname=reduction_tensor_name,
                                             mode=self.mode, mode_step=self.mode_steps[self.mode])
                        s_col.add_reduction_tensor_name(tensor_name)
                    return
                else:
                    tensor_value = make_numpy_array(tensor_value)
                    self.writer.write_tensor(tdata=tensor_value, tname=tensor_name,
                                             mode=self.mode, mode_step=self.mode_steps[self.mode])
                    return

    # This function is "applied" to every child in the block. This function in turn
    # registers the forward hook to each block. It helps logging the input output tensors
    # of that block.
    def _recursive_apply(self, block):
        block.register_forward_hook(self.forward_hook)

    # This function registers the forward hook. If user wants to register the hook
    # for every child in the given block, then the function calls "apply" API for
    # registration of the hook.
    # The hook is registered recursively, if user has specified the collections that are more than
    # the default collectors viz. gradients, weight and bias
    def register_hook(self, block):
        self.is_recursive=True
        if not isinstance(block, mx.gluon.Block):
            logger.error("The given block type {0} is not "
                         "currently supported by Tornasole Hook"
                         .format(block.__class__.__name__))
            return
        block.register_forward_pre_hook(self.forward_pre_hook)
        if self.is_recursive:
            block.apply(self._recursive_apply)
        else:
            block.register_forward_hook(self.forward_hook)

    @staticmethod
    def clean_tag(name):
        if name is not None:
            new_name = INVALID_TAG_CHARACTERS.sub('_', name)
            new_name = new_name.lstrip('/')  # Remove leading slashes
            if new_name != name:
                logging.warning('Summary name %s is illegal; using %s instead.', name, new_name)
                name = new_name
        return name
      
