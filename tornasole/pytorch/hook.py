import torch
from tornasole.core.writer import FileWriter
from tornasole.core.save_config import SaveConfig
from tornasole.core.save_manager import SaveManager
from tornasole.core.modes import ModeKeys, ALLOWED_MODES
from tornasole.core.logger import get_logger
from tornasole.core.hook_utils import verify_and_get_out_dir
from tornasole.core.reductions import get_reduction_tensor_name
from tornasole.core.json_config import create_hook_from_json_config
from tornasole.pytorch.torch_collection import get_collection_manager, get_collection
from tornasole.pytorch.util import get_aggregated_data, make_numpy_array
from tornasole.core.access_layer.utils import training_has_ended
from tornasole.core.collection_manager import COLLECTIONS_FILE_NAME

import re as _re
import logging
import os

logger = get_logger()
import atexit

INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')
DEFAULT_WORKER_NAME = 'worker0'
INPUT_TENSOR_SUFFIX = '_input_'
OUTPUT_TENSOR_SUFFIX = '_output'
GRADIENT_PREFIX = 'gradient/'
DEFAULT_INCLUDE_COLLECTIONS = ['weights', 'bias', 'gradients', 'default']

class TornasoleHook:
    def __init__(self,
                 out_dir=None,
                 dry_run=False,
                 worker=DEFAULT_WORKER_NAME,
                 reduction_config=None,
                 save_config=None,
                 include_regex=None,
                 include_collections=DEFAULT_INCLUDE_COLLECTIONS,
                 save_all=False):
        self.out_dir = verify_and_get_out_dir(out_dir)

        self.include_collections = include_collections

        self.dry_run = dry_run
        self.worker = worker

        self.mode = ModeKeys.GLOBAL
        self.mode_steps = {ModeKeys.GLOBAL: -1}

        self.reduction_config = reduction_config
        self.step = -1
        self.is_recursive = False
        self.export_only_once = True
        self.last_saved_step = -1
        self.writer = None
        self._initialize_collectors(save_all, include_regex)

        # dictionary of collections that need to be saved in a particular step.
        self.collections_in_this_step = None
        # mapping of module objects to their names, useful in forward hook for logging input/output of modules
        self.module_maps = dict()
        self.exported_collection = False

        atexit.register(self.cleanup)

        if save_config is None:
            save_config = SaveConfig()
        self.save_manager = SaveManager(collection_manager=get_collection_manager(),
                                        include_collections_names=self.include_collections,
                                        default_save_config=save_config,
                                        default_reduction_config=reduction_config)
        self.prepared_save_manager = False
        logger.info('Saving to {}'.format(self.out_dir))

    @classmethod
    def hook_from_config(cls):
        return create_hook_from_json_config(cls, get_collection_manager(), DEFAULT_INCLUDE_COLLECTIONS)

    def _initialize_collectors(self, save_all, include_regex):
        # If user has provided any include_regex, add them to a default collection.
        if include_regex is not None:
            get_collection('default').include(include_regex)
            if 'default' not in self.include_collections:
                self.include_collections.append('default')
        # If save all is set, create a collector that can save all the tensors
        if save_all:
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
        if not self.exported_collection:
            get_collection_manager().export_manager(os.path.join(self.out_dir, COLLECTIONS_FILE_NAME))
        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None:
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
    def forward_pre_hook(self, module, input):
        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None:
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
            self.writer = FileWriter(trial_dir=self.out_dir,
                                     step=self.step,
                                     worker=self.worker)
            module_name = module._get_name()
            params = module.named_parameters()
            for name, param in params:
                pname = module_name + '_' + name
                logger.debug("Processing the global step {0} for parameter {1}".format(self.step, pname))
                self.log_tensor(tensor_name=pname, tensor_value=param.data)

        if self.last_saved_step != -1 and not self.exported_collection:
            get_collection_manager().export_manager(os.path.join(self.out_dir, COLLECTIONS_FILE_NAME))
            self.exported_collection = True
        # self.last_block = block

    # This hook is invoked by trainer after running the forward pass.
    def forward_hook(self, module, input, output):
        if not self.collections_in_this_step:
            logging.debug("Skipping the global step {0}".format(self.step))
            return

        module_name = self.module_maps[module]
        logger.debug("Processing the global step {0} for module {1}".format(self.step, module_name))

        # Output input tensor
        self.log_inputs_to_module(module_name, input)

        # Output output tensors
        self.log_outputs_of_module(module_name, output)
        self.last_saved_step = self.step

    def backward_hook(self, tname):
        # Helper function that has access to the parameter name via the scope in which it's defined.
        def back(grad):
            if self._process_step():
                if grad is not None:
                    logger.debug("Processing the backward step {0} for {1}".format(self.step, tname))
                    self.log_tensor(tensor_name=GRADIENT_PREFIX + tname, tensor_value=grad)
        return back

    def log_module(self, module_name, var, suffix, idx):
        if var.__class__.__name__ is "Tensor":
            self.log_tensor(tensor_name=module_name + suffix + str(idx), tensor_value=var)
            return idx + 1
        elif isinstance(var, tuple) or isinstance(var, list):
            for val in var:
                idx = self.log_module(module_name, val, suffix, idx)
        else:
            logger.warning("var is not Tensor or list of Tensors, module_name:{} {}".format(module_name, var.__class__.__name__))

    def log_inputs_to_module(self, module_name, input):
        idx = 0
        self.log_module(module_name, input, INPUT_TENSOR_SUFFIX, idx)

    def log_outputs_of_module(self, module_name, output):
        idx = 0
        self.log_module(module_name, output, OUTPUT_TENSOR_SUFFIX, idx)

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
                        s_col.reduction_tensor_names.add(reduction_tensor_name)
                    return
                else:
                    tensor_value = make_numpy_array(tensor_value)
                    self.writer.write_tensor(tdata=tensor_value, tname=tensor_name,
                                             mode=self.mode, mode_step=self.mode_steps[self.mode])
                    return
    # TODO: remove? not being used anywhere
    def close_log(self):
        if self._process_step():
            return
        self.writer.close()

    # This function is "applied" to every child in the block. This function in turn
    # registers the forward hook to each module. It helps logging the input output tensors
    # of that module.

    def _recursive_apply(self, module):
        module.register_forward_hook(self.forward_hook)

    def _backward_apply(self, module):
        params = module.named_parameters()
        for name, param in params:
            pname = module._get_name() + '_' + name
            param.register_hook(self.backward_hook(pname))

    # This function registers the forward hook. If user wants to register the hook
    # for every child in the given block, then the function calls "apply" API for
    # registration of the hook.
    # The hook is registered recursively, if user has specified the collections that are more than
    # the default collectors viz. gradients, weight and bias
    def register_hook(self, module):
        if not isinstance(module, torch.nn.Module):
            logger.error("The given module type {0} is not currently supported by Tornasole Hook".format(
                module.__class__.__name__))
            return
        module.register_forward_pre_hook(self.forward_pre_hook)

        for layer in list(module.named_modules()):
            self.module_maps[layer[1]] = layer[0]
        self.module_maps[module] = module._get_name()
        module.apply(self._recursive_apply)
        self._backward_apply(module)

    @staticmethod
    def clean_tag(name):
        if name is not None:
            new_name = INVALID_TAG_CHARACTERS.sub('_', name)
            new_name = new_name.lstrip('/')  # Remove leading slashes
            if new_name != name:
                logging.warning('Summary name %s is illegal; using %s instead.', name, new_name)
                name = new_name
        return name
