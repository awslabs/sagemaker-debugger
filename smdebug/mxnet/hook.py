# Third Party
import mxnet as mx

# First Party
from smdebug.core.collection import DEFAULT_MXNET_COLLECTIONS, CollectionKeys
from smdebug.core.hook import CallbackHook
from smdebug.core.json_config import DEFAULT_WORKER_NAME
from smdebug.mxnet.collection import CollectionManager
from smdebug.mxnet.graph import _net2pb
from smdebug.mxnet.singleton_utils import set_hook
from smdebug.mxnet.utils import get_reduction_of_data, make_numpy_array

DEFAULT_INCLUDE_COLLECTIONS = [CollectionKeys.LOSSES]

COLLECTIONS_NOT_REQUIRING_RECURSIVE_HOOK = [
    CollectionKeys.WEIGHTS,
    CollectionKeys.BIASES,
    CollectionKeys.GRADIENTS,
    CollectionKeys.LOSSES,
]


class Hook(CallbackHook):
    def __init__(
        self,
        out_dir=None,
        export_tensorboard=False,
        tensorboard_dir=None,
        dry_run=False,
        reduction_config=None,
        save_config=None,
        include_regex=None,
        include_collections=None,
        save_all=False,
        include_workers="one",
    ):
        collection_manager = CollectionManager()
        super().__init__(
            collection_manager=collection_manager,
            default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
            data_type_name=mx.ndarray.NDArray.__name__,
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
        self.last_block = None

        self.model = None
        self.exported_model = False
        # Keep the set of blocks to which this hook is registered. The blocks include loss blocks as well.
        self.registered_blocks = set()
        self.worker = self._get_worker_name()
        set_hook(self)

    def _get_worker_name(self):
        try:
            import horovod.mxnet as hvd

            if hvd.size():
                return f"worker_{hvd.rank()}"
        except (ModuleNotFoundError, ValueError, ImportError):
            pass

        return DEFAULT_WORKER_NAME

    def _get_num_workers(self):
        try:
            import horovod.mxnet as hvd

            if hvd.size():
                return hvd.size()
        except (ModuleNotFoundError, ValueError, ImportError):
            pass
        return 1

    def _cleanup(self):
        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None and self.last_block is not None:
            self._log_params(self.last_block)
        if self.exported_model is False:
            self._export_model()
        super()._cleanup()

    def _log_params(self, block):
        params = block.collect_params().values()
        for param in params:
            self._log_param(param)

    def _log_param(self, param):
        try:
            self._save_for_tensor(
                tensor_name=param.name, tensor_value=param.data(param.list_ctx()[0])
            )
            # If Gradient for this param is available
            if param.grad_req != "null":
                self._save_for_tensor(
                    tensor_name=self.GRADIENT_PREFIX + param.name,
                    tensor_value=param.grad(param.list_ctx()[0]),
                )
        except RuntimeError as e:
            self.logger.warning(
                f"Could not log parameter {param.name} due to the mxnet exception: {e}"
            )

    def _export_model(self):
        if self.model is not None:
            try:
                tb_writer = self._maybe_get_tb_writer()
                if tb_writer:
                    tb_writer.write_graph(_net2pb(self.model))
            except (RuntimeError, TypeError) as e:
                self.logger.warning(
                    f"Could not export model graph for tensorboard "
                    f"due to the mxnet exception: {e}"
                )

    def _get_default_collections(self):
        return DEFAULT_MXNET_COLLECTIONS

    # This hook is invoked by trainer prior to running the forward pass.
    def forward_pre_hook(self, block, inputs):
        if self.writer is not None:
            # Write the params and gradients of the
            # past step if the writer is still available.
            self._log_params(block)
            self._close_writers()
        self._close_tb_writer()

        if not self.prepared_collections:
            # at this point we need all collections to be ready
            # this may not be the case at creation of hook
            # as user's code after hook might add collections
            self._prepare_collections()
            self.prepared_collections = True

        self._increment_step()

        if self._get_collections_to_save_for_step():
            self._initialize_writers()

        if self.exported_model is False:
            self._export_model()
            self.exported_model = True

        if self.last_saved_step is not None and not self.exported_collections:
            self.export_collections()
            self.exported_collections = True

        self.last_block = block

    # This hook is invoked by trainer after running the forward pass.
    def forward_hook(self, block, inputs, outputs):
        if not self._get_collections_to_save_for_step():
            return

        block_name = block.name
        # This overwhelms the logs; turn back on if you really need it
        # logger.debug("Processing the global step {0} for block {1}".format(self.step, block_name))

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
        # Check if the hook is already registered for this block.
        if block in self.registered_blocks:
            self.logger.warning(f"The hook is already registered to block {block.name}")
            return
        block.register_forward_hook(self.forward_hook)
        self.registered_blocks.add(block)

    def _is_recursive_needed(self):
        collections_to_save = self.include_collections

        # Check if default collection has a regex associated with it.
        # If it does we would need to apply hook recursively.
        if (
            len(self.collection_manager.get(CollectionKeys.DEFAULT).include_regex) != 0
            and CollectionKeys.DEFAULT in collections_to_save
        ):
            return True

        # Get the collections that are to be saved but are not part of default collections
        # We will need to apply hook recursively to get tensors specified in those collections.
        extra_coll = [
            value
            for value in collections_to_save
            if value not in COLLECTIONS_NOT_REQUIRING_RECURSIVE_HOOK
        ]

        # extra_coll contains the collections that are not part of default collections.
        return len(extra_coll) != 0

    def register_hook(self, block):
        # for compatibility with ZCC patches which call this
        self.register_block(block)

    def register_block(self, block):
        """
        This function registers the forward hook. If user wants to register the hook
        for every child in the given block, then the function calls "apply" API for
        registration of the hook.
        The hook is registered recursively, if user has specified the collections that are more than
        the default collectors viz. gradients, weight and bias
        """

        if not isinstance(block, mx.gluon.Block):
            self.logger.error(f"The given block type {block.__class__.__name__} is unsupported.")
            return

        # Check if the hook is already registered for this block.
        if block in self.registered_blocks:
            self.logger.warning(f"The hook is already registered to block {block.name}")
            return

        # Skip the forward pre hook for the Loss blocks.
        if isinstance(block, mx.gluon.loss.Loss):
            self.logger.info(f"Registering hook for block {block.name}")
            block.register_forward_hook(self.forward_hook)
            self.registered_blocks.add(block)
            return
        else:
            self.model = block

        is_recursive = self._is_recursive_needed()
        block.register_forward_pre_hook(self.forward_pre_hook)
        if is_recursive is True:
            block.apply(self._recursive_apply)
        else:
            block.register_forward_hook(self.forward_hook)
            self.registered_blocks.add(block)

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        return get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs)

    @staticmethod
    def _make_numpy_array(tensor_value):
        return make_numpy_array(tensor_value)
