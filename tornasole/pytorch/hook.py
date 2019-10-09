import importlib
import torch
import torch.distributed as dist
import logging
from tornasole.core.hook import CallbackHook
from tornasole.core.collection import CollectionKeys
from tornasole.core.logger import get_logger
from tornasole.core.json_config import create_hook_from_json_config
from tornasole.pytorch.collection import get_collection_manager
from tornasole.pytorch.utils import get_reduction_of_data, make_numpy_array
from tornasole.core.json_config import TORNASOLE_CONFIG_DEFAULT_WORKER_NAME

logger = get_logger()

DEFAULT_INCLUDE_COLLECTIONS = [
    CollectionKeys.WEIGHTS,
    CollectionKeys.BIASES,
    CollectionKeys.GRADIENTS,
    CollectionKeys.DEFAULT,
    CollectionKeys.LOSSES,
]


class TornasoleHook(CallbackHook):
    def __init__(self,
                 out_dir=None,
                 dry_run=False,
                 reduction_config=None,
                 save_config=None,
                 include_regex=None,
                 include_collections=None,
                 save_all=False):

        super().__init__(
                collection_manager=get_collection_manager(),
                default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
                data_type_name=torch.Tensor.__name__,
                out_dir=out_dir,
                dry_run=dry_run,
                reduction_config=reduction_config,
                save_config=save_config,
                include_regex=include_regex,
                include_collections=include_collections,
                save_all=save_all)
        # mapping of module objects to their names,
        # useful in forward hook for logging input/output of modules
        self.module_maps = dict()

    def get_num_workers(self):
        """Check horovod and torch.distributed."""
        # Try torch.distributed
        # torch.distributed is empty on Mac on Torch <= 1.2
        if hasattr(dist, 'is_initialized') and dist.is_initialized():
            return torch.distributed.get_world_size()
        # Try horovod
        else:
            try:
                import horovod.torch as hvd
                if hvd.size():
                    return hvd.size()
            except (ModuleNotFoundError, ValueError, ImportError):
                pass
        # Return default
        return 1

    def get_worker_name(self):
        """Check horovod and torch.distributed."""
        # Try torch.distributed
        # torch.distributed is empty on Mac on Torch <= 1.2
        if hasattr(dist, 'is_initialized') and dist.is_initialized():
            return f"worker_{dist.get_rank()}"
        # Try horovod
        else:
            try:
                import horovod.torch as hvd
                if hvd.size():
                    return f"worker_{hvd.rank()}"
            except (ModuleNotFoundError, ValueError, ImportError):
                pass
        # Return default
        return TORNASOLE_CONFIG_DEFAULT_WORKER_NAME

    @classmethod
    def hook_from_config(cls):
        return create_hook_from_json_config(cls, get_collection_manager())

    def log_params(self, module):
        module_name = module._get_name()
        params = module.named_parameters()
        for name, param in params:
            pname = module_name + '_' + name
            # This overwhelms the logs; turn back on if you really need it
            # self.logger.debug(
                # "Processing the global step {0} for parameter {1}".format(self.step, pname))
            self._write_tensor(tensor_name=pname, tensor_value=param.data)

    # This hook is invoked by trainer prior to running the forward pass.
    def forward_pre_hook(self, module, inputs):
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
            self.log_params(module)

        if self.last_saved_step is not None and not self.exported_collections:
            self.export_collections()
            self.exported_collections = True

    # This hook is invoked by trainer after running the forward pass.
    def forward_hook(self, module, inputs, outputs):
        if self.collections_in_this_step is None:
            logging.debug("Skipping the global step {0}".format(self.step))
            return

        module_name = self.module_maps[module]
        # This overwhelms the logs; turn back on if you really need it
        # logger.debug("Processing the global step {0} for module {1}".format(self.step, module_name))

        # Output input tensor
        self._write_inputs(module_name, inputs)

        # Output output tensors
        self._write_outputs(module_name, outputs)
        self.last_saved_step = self.step

    def backward_hook(self, tname):
        # Helper function that has access to the parameter name via the scope in which it's defined.
        def back(grad):
            if self._process_step():
                if grad is not None:
                    logger.debug("Processing the backward step {0} for {1}".format(self.step, tname))
                    self._write_tensor(tensor_name=self.GRADIENT_PREFIX + tname, tensor_value=grad)
        return back

    def _backward_apply(self, module):
        """Apply the function `self.backward_hook` as a callback to each parameter in `module.

        This will capture the gradients.
        """
        params = module.named_parameters()
        for name, param in params:
            pname = module._get_name() + '_' + name
            param.register_hook(self.backward_hook(pname))

    def closure_for_registering_forward_hook(self, module):
        """Lambda functions don't work here."""
        module.register_forward_hook(self.forward_hook)

    def register_hook(self, module):
        """
        This function registers the forward hook. If user wants to register the hook
        for every child in the given block, then the function calls "apply" API for
        registration of the hook.
        The hook is registered recursively for all blocks.
        """
        # Typechecking
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"Module type {module.__class__.__name__} must be type torch.nn.Module")

        # Create a mapping from modules to their names
        for name, submodule in module.named_modules():
            assert submodule not in self.module_maps, f"Don't register module={module} twice"
            self.module_maps[submodule] = name
        self.module_maps[module] = module._get_name()

        # Use `forward_pre_hook` for the entire net
        module.register_forward_pre_hook(self.forward_pre_hook)

        # Set `self.forward_hook` as a callback for each submodule/layer.
        # `module.apply(fn)` calls fn for each submodule in module.children()
        module.apply(self.closure_for_registering_forward_hook)

        # Capture the gradient for each parameter in the net
        self._backward_apply(module)

    def register_loss(self, loss_module):
        """Register something like `criterion = nn.CrossEntropyLoss()`."""
        # Typechecking
        assert isinstance(loss_module, torch.nn.modules.loss._Loss), (
            f"loss_module={loss_module} must be subclass of `torch.nn.modules.loss._Loss`, "
            f"but has class hierarchy {type.mro(type(loss_module))}"
        )
        # Register the module in self.module_maps
        name = loss_module._get_name()
        self.module_maps[loss_module] = name
        # Add a callback to the forward pass
        loss_module.register_forward_hook(self.forward_hook)


    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        return get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs)

    @staticmethod
    def _make_numpy_array(tensor_value):
        return make_numpy_array(tensor_value)
