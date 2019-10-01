import logging
import mxnet as mx
from tornasole.core.hook import CallbackHook
from tornasole.core.logger import get_logger
from tornasole.core.json_config import TORNASOLE_CONFIG_DEFAULT_WORKER_NAME, create_hook_from_json_config
from tornasole.core.collection import CollectionKeys
from tornasole.mxnet.mxnet_collection import get_collection_manager
from tornasole.mxnet.utils import get_reduction_of_data, make_numpy_array

logger = get_logger()

DEFAULT_INCLUDE_COLLECTIONS = [CollectionKeys.LOSSES]

COLLECTIONS_NOT_REQUIRING_RECURSIVE_HOOK = [
    CollectionKeys.WEIGHTS,
    CollectionKeys.BIASES,
    CollectionKeys.GRADIENTS,
    CollectionKeys.LOSSES
]


class TornasoleHook(CallbackHook):
    def __init__(self,
                 out_dir=None,
                 dry_run=False,
                 worker=TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
                 reduction_config=None,
                 save_config=None,
                 include_regex=None,
                 include_collections=None,
                 save_all=False):
        super().__init__(
                collection_manager=get_collection_manager(),
                default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
                data_type_name=mx.ndarray.NDArray.__name__,
                out_dir=out_dir,
                dry_run=dry_run,
                worker=worker,
                reduction_config=reduction_config,
                save_config=save_config,
                include_regex=include_regex,
                include_collections=include_collections,
                save_all=save_all)
        # We would like to collect loss collection
        # even if user does not specify any collections
        if CollectionKeys.LOSSES not in self.include_collections:
            self.include_collections.append(CollectionKeys.LOSSES)
        self.last_block = None

    @classmethod
    def hook_from_config(cls):
        return create_hook_from_json_config(cls, get_collection_manager())

    def _cleanup(self):
        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None and self.last_block is not None:
            self.log_params(self.last_block)
        super()._cleanup()

    def log_params(self, block):
        params = block.collect_params().values()
        for param in params:
            self.log_param(param)

    def log_param(self, param):
        self._write_tensor(tensor_name=param.name, tensor_value=param.data(param.list_ctx()[0]))
        # If Gradient for this param is available
        if param.grad_req != 'null':
            self._write_tensor(tensor_name=self.GRADIENT_PREFIX + param.name,
                               tensor_value=param.grad(param.list_ctx()[0]))

    # This hook is invoked by trainer prior to running the forward pass.
    def forward_pre_hook(self, block, inputs):
        if self.writer is not None:
            # Write the params and gradients of the
            # past step if the writer is still available.
            self.log_params(block)
            self._flush_and_close_writer()

        if not self.prepared_collections:
            # at this point we need all collections to be ready
            # this may not be the case at creation of hook
            # as user's code after hook might add collections
            self._prepare_collections()
            self.prepared_collections = True

        self._increment_step()

        if self._process_step():
            self._initialize_writer()

        if self.last_saved_step is not None and not self.exported_collections:
            self.export_collections()
            self.exported_collections = True
        self.last_block = block

    # This hook is invoked by trainer after running the forward pass.
    def forward_hook(self, block, inputs, outputs):
        if self.collections_in_this_step is None:
            logging.debug("Skipping the global step {0}".format(self.step))
            return

        block_name = block.name
        logger.debug("Processing the global step {0} for block {1}".format(self.step, block_name))

        # Output input tensor
        self._write_inputs(block_name, inputs)

        # Output output tensors
        self._write_outputs(block_name, outputs)
        self.last_saved_step = self.step

    def _recursive_apply(self, block):
        """
        This function is "applied" to every child in the block. This function in turn
        registers the forward hook to each module. It helps logging the input output tensors
        of that module.
        """
        block.register_forward_hook(self.forward_hook)

    def _is_recursive_needed(self):
        collections_to_save = self.include_collections

        # Check if default collection has a regex associated with it.
        # If it does we would need to apply hook recursively.
        if len(self.collection_manager.get(CollectionKeys.DEFAULT).get_include_regex()) != 0 \
                and CollectionKeys.DEFAULT in collections_to_save:
            return True

        # Get the collections that are to be saved but are not part of default collections
        # We will need to apply hook recursively to get tensors specified in those collections.
        extra_coll = [value for value in collections_to_save if value not in COLLECTIONS_NOT_REQUIRING_RECURSIVE_HOOK]

        # extra_coll contains the collections that are not part of default collections.
        return len(extra_coll) != 0

    def register_hook(self, block):
        """
        This function registers the forward hook. If user wants to register the hook
        for every child in the given block, then the function calls "apply" API for
        registration of the hook.
        The hook is registered recursively, if user has specified the collections that are more than
        the default collectors viz. gradients, weight and bias
        """
        if not isinstance(block, mx.gluon.Block):
            logger.error("The given block type {0} is not "
                         "currently supported by Tornasole Hook"
                         .format(block.__class__.__name__))
            return

        # Skip the forward pre hook for the Loss blocks.
        if isinstance(block, mx.gluon.loss.Loss):
            logger.info("Registering hook for block {0}".format(block.name))
            block.register_forward_hook(self.forward_hook)
            return

        is_recursive = self._is_recursive_needed()
        block.register_forward_pre_hook(self.forward_pre_hook)
        if is_recursive is True:
            block.apply(self._recursive_apply)
        else:
            block.register_forward_hook(self.forward_hook)

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        return get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs)

    @staticmethod
    def _make_numpy_array(tensor_value):
        return make_numpy_array(tensor_value)
