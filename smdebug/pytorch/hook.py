# Standard Library
import time

# Third Party
import torch
import torch.distributed as dist

# First Party
from smdebug.core.collection import DEFAULT_PYTORCH_COLLECTIONS, CollectionKeys
from smdebug.core.hook import CallbackHook
from smdebug.core.json_config import DEFAULT_WORKER_NAME
from smdebug.profiler.hvd_trace_file_rotation import HvdTraceFileRotation
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS
from smdebug.pytorch.collection import CollectionManager
from smdebug.pytorch.singleton_utils import set_hook
from smdebug.pytorch.utils import get_reduction_of_data, make_numpy_array

DEFAULT_INCLUDE_COLLECTIONS = [CollectionKeys.LOSSES]


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
            data_type_name=torch.Tensor.__name__,
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
        # mapping of module objects to their names,
        # useful in forward hook for logging input/output of modules
        self.module_set = set()

        self.has_registered_module = False
        self.has_registered_loss_module = False
        self.worker = self._get_worker_name()

        # Only the chief worker will read the Horovod timeline file
        # if HOROVOD_TIMELINE is a valid file and SM Profiler is enabled
        if not self.hvd_reader and self.worker == self.chief_worker:
            self.hvd_reader = HvdTraceFileRotation(self.profiler_config_parser)

        set_hook(self)

        self.autograd_profiler_enabled = False
        self.profiler = (
            torch.autograd.ProfilerState.CUDA
            if torch.cuda.is_available()
            else torch.autograd.ProfilerState.CPU
        )

    def _get_num_workers(self):
        """Check horovod and torch.distributed."""
        # Try torch.distributed
        # torch.distributed is empty on Mac on Torch <= 1.2
        if hasattr(dist, "is_initialized") and dist.is_initialized():
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

    def _get_worker_name(self):
        """Check horovod and torch.distributed."""
        # Try torch.distributed
        # torch.distributed is empty on Mac on Torch <= 1.2
        if hasattr(dist, "is_initialized") and dist.is_initialized():
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
        return DEFAULT_WORKER_NAME

    def _log_params(self, module):
        module_name = module._get_name()
        params = module.named_parameters()
        for name, param in params:
            pname = module_name + "_" + name
            # This overwhelms the logs; turn back on if you really need it
            # self.logger.debug(
            # "Processing the global step {0} for parameter {1}".format(self.step, pname))
            self._save_for_tensor(tensor_name=pname, tensor_value=param.data)

    def _export_model(self):
        pass

    def _get_default_collections(self):
        return DEFAULT_PYTORCH_COLLECTIONS

    def _prepare_collections(self):
        for coll in self.collection_manager.collections.values():
            for m, (include_inputs, include_outputs) in coll.modules.items():
                module_name = m._module_name
                if include_inputs:
                    coll.include(module_name + "_input_")
                if include_outputs:
                    coll.include(module_name + "_output_")
        super()._prepare_collections()

    # This hook is invoked by trainer prior to running the forward pass.
    def forward_pre_hook(self, module, inputs):
        if self.profiler_config_parser.can_start_detailed_profiling(self.step):
            if not self.autograd_profiler_enabled:
                torch.autograd._enable_profiler(torch.autograd.ProfilerConfig(self.profiler, False))
                self.autograd_profiler_enabled = True
                self.start_profiler = time.time() * CONVERT_TO_MICROSECS
        else:
            if self.autograd_profiler_enabled:
                records = torch.autograd._disable_profiler()
                self.function_events = torch.autograd.profiler.EventList(
                    torch.autograd.profiler.parse_cpu_trace(records), use_cuda=True
                )
                for index, event in enumerate(self.function_events):
                    self.record_trace_events(
                        phase="X",
                        op_name=event.name,
                        timestamp=(event.cpu_interval.start + self.start_profiler)
                        / float(CONVERT_TO_MICROSECS),
                        duration=event.cpu_interval.elapsed_us() / float(CONVERT_TO_MICROSECS),
                        tid=event.thread,
                        step_num=self.step,
                    )
                for k in event.kernels:
                    self.record_trace_events(
                        op_name=k.name,
                        phase="X",
                        timestamp=(k.interval.start + self.start_profiler)
                        / float(CONVERT_TO_MICROSECS),
                        duration=k.interval.elapsed_us() / float(CONVERT_TO_MICROSECS),
                        tid=k.device,
                        step_num=self.step,
                    )
                self.autograd_profiler_enabled = False
        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None:
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
            self._log_params(module)

        if self.last_saved_step is not None and not self.exported_collections:
            self.export_collections()
            self.exported_collections = True

    def record_tensor_value(self, tensor_name: str, tensor_value: torch.Tensor) -> None:
        """Used for registering functional directly, such as F.mse_loss()."""
        assert isinstance(
            tensor_value, torch.Tensor
        ), f"tensor_value={tensor_value} must be torch.Tensor"

        self._write_outputs(tensor_name, tensor_value)

    # This hook is invoked by trainer after running the forward pass.
    def forward_hook(self, module, inputs, outputs):
        if not self._get_collections_to_save_for_step():
            return

        module_name = module._module_name
        # This overwhelms the logs; turn back on if you really need it
        # logger.debug("Processing the global step {0} for module {1}".format(self.step, module_name))

        # Output input tensor
        self._write_inputs(module_name, inputs)

        # Output output tensors
        self._write_outputs(module_name, outputs)
        self.last_saved_step = self.step

    def backward_hook(self, tname):
        # Helper function that has access to the parameter name via
        # the scope in which it's defined.
        def back(grad):
            if self._get_collections_to_save_for_step():
                if grad is not None:
                    # self.logger.debug(f"Processing the backward step " f"{self.step} for {tname}")
                    self._save_for_tensor(self.GRADIENT_PREFIX + tname, grad)

        return back

    def _backward_apply(self, module):
        """Apply the function `self.backward_hook` as a callback to each parameter in `module.

        This will capture the gradients.
        """
        params = module.named_parameters()
        for name, param in params:
            pname = module._get_name() + "_" + name
            if param.requires_grad:
                param.register_hook(self.backward_hook(pname))

    def _closure_for_registering_forward_hook(self, module):
        """Lambda functions don't work here."""
        module.register_forward_hook(self.forward_hook)

    def register_hook(self, module):
        # for compatibility with ZCC patches which call this
        self.register_module(module)

    def register_module(self, module):
        """
        This function registers the forward hook. If user wants to register the hook
        for every child in the given block, then the function calls "apply" API for
        registration of the hook.
        The hook is registered recursively for all blocks.
        """
        # Typechecking
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"Module type {module.__class__.__name__} must be type torch.nn.Module"
            )

        # Create an attribute and store the module name in the object
        # So that it is available in the forward hook.

        for name, submodule in module.named_modules():
            assert submodule not in self.module_set, f"Don't register module={module} twice"
            submodule._module_name = name
            self.module_set.add(submodule)
        module._module_name = module._get_name()
        self.module_set.add(module)

        # Use `forward_pre_hook` for the entire net
        module.register_forward_pre_hook(self.forward_pre_hook)

        # Set `self.forward_hook` as a callback for each submodule/layer.
        # `module.apply(fn)` calls fn for each submodule in module.children()
        module.apply(self._closure_for_registering_forward_hook)

        # Capture the gradient for each parameter in the net
        self._backward_apply(module)

        self.has_registered_module = True

    def register_loss(self, loss_module):
        """Register something like `criterion = nn.CrossEntropyLoss()`."""
        # Typechecking
        assert isinstance(loss_module, torch.nn.modules.loss._Loss), (
            f"loss_module={loss_module} must be subclass of `torch.nn.modules.loss._Loss`, "
            f"but has class hierarchy {type.mro(type(loss_module))}"
        )
        loss_module._module_name = loss_module._get_name()
        self.module_set.add(loss_module)
        # Add a callback to the forward pass
        loss_module.register_forward_hook(self.forward_hook)
        self.has_registered_loss_module = True

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        return get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs)

    @staticmethod
    def _make_numpy_array(tensor_value):
        return make_numpy_array(tensor_value)
