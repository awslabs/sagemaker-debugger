# Standard Library
import atexit
import os
import time

# Third Party
import torch
import torch.distributed as dist

# First Party
from smdebug.core.collection import DEFAULT_PYTORCH_COLLECTIONS, CollectionKeys
from smdebug.core.hook import CallbackHook
from smdebug.core.json_config import DEFAULT_WORKER_NAME
from smdebug.core.utils import check_smdataparallel_env, make_numpy_array
from smdebug.profiler.hvd_trace_file_rotation import HvdTraceFileRotation
from smdebug.profiler.profiler_config_parser import MetricsCategory, ProfilerConfigParser
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS
from smdebug.profiler.python_profile_utils import StepPhase, mode_keys_to_python_profile_mode
from smdebug.profiler.python_profiler import PythonProfiler
from smdebug.profiler.utils import start_smdataparallel_profiler, stop_smdataparallel_profiler
from smdebug.pytorch.collection import CollectionManager
from smdebug.pytorch.singleton_utils import set_hook
from smdebug.pytorch.utils import get_reduction_of_data, is_pt_1_5, is_pt_1_6, is_pt_1_7, is_pt_1_8

# smdistributed.dataparallel should be invoked via `mpirun`.
# It supports EC2 machines with 8 GPUs per machine.
smdataparallel = None
if check_smdataparallel_env():
    try:
        import smdistributed.dataparallel.torch.distributed as smdataparallel
    except ImportError:
        pass


DEFAULT_INCLUDE_COLLECTIONS = [CollectionKeys.LOSSES]


python_profiler = None

# Enable python profiling if profiling is enabled.
profiler_config_parser = ProfilerConfigParser()
if profiler_config_parser.profiling_enabled:
    config = profiler_config_parser.config
    if config.python_profiling_config.is_enabled():
        python_profiler = PythonProfiler.get_python_profiler(config, "pytorch")
        python_profiler.start_profiling(StepPhase.START)


class Hook(CallbackHook):
    """
    The _TraceEventData is similar to a structure that contains the event data to be written in the event file.
    It contains the following metadata:
    training_phase such as "X", "M", "I" etc.
    start time, duration and end time in microseconds
    pid is the process id that wants to record this event.
    and event arguments.
    """

    class _TraceEventData:
        def __init__(self, phase, op_name, start_time, dur, **kwargs):
            self.training_phase = phase
            self.end_time = start_time + dur
            self.start_time = start_time
            self.op_name = op_name
            self.kwargs = kwargs

        def update_end_time(self, end_time=time.time()):
            self.end_time = end_time

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
            profiler_config_parser=profiler_config_parser,
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
        self.parent_forward_event = None
        self.parent_backward_event = None
        self.step_event = None
        self.forward_modules_profile_stats = []
        self.backward_modules_profile_stats = []
        self.first_forward_submodule_name = None
        self.autograd_profiler_enabled = False
        self.profiler = (
            torch.autograd.ProfilerState.CUDA
            if torch.cuda.is_available()
            else torch.autograd.ProfilerState.CPU
        )
        self.use_cuda = torch.cuda.is_available()
        if python_profiler:
            atexit.register(python_profiler.stop_profiling, StepPhase.END)

    def log_trace_event(self, event):
        self.record_trace_events(
            training_phase=event.training_phase,
            op_name=event.op_name,
            phase="X",
            timestamp=event.start_time,
            duration=event.end_time - event.start_time,
            **(event.kwargs),
        )

    def reset_forward_module_profile_stats(self):
        self.parent_forward_event = None
        self.forward_modules_profile_stats = []

    def reset_backward_module_profile_stats(self):
        self.parent_backward_event = None
        self.backward_modules_profile_stats = []

    def log_outstanding_timeline_metrics(self):
        self.log_outstanding_forward_stats_and_reset()
        self.log_outstanding_backward_stats_and_reset()

    def log_outstanding_forward_stats_and_reset(self, log_step_event=True):
        # we need to skip the last event for submodules because that is usually the parent event
        # and already recorded above
        for i in range(len(self.forward_modules_profile_stats) - 1):
            event = self.log_trace_event(self.forward_modules_profile_stats[i])

        if self.parent_forward_event:
            self.log_trace_event(self.parent_forward_event)
        if self.step_event and log_step_event is True:
            self.log_trace_event(self.step_event)
            self.step_event = None

        self.reset_forward_module_profile_stats()

    def log_outstanding_backward_stats_and_reset(self):
        for i in range(len(self.backward_modules_profile_stats)):
            event = self.log_trace_event(self.backward_modules_profile_stats[i])

        if self.parent_backward_event:
            self.log_trace_event(self.parent_backward_event)
        self.reset_backward_module_profile_stats()

    def _get_num_workers(self):
        """Check horovod, smdataparallel, and torch.distributed."""
        # Try torch.distributed
        # torch.distributed is empty on Mac on Torch <= 1.2
        if hasattr(dist, "is_initialized") and dist.is_initialized():
            return torch.distributed.get_world_size()
        else:
            # Try horovod
            try:
                import horovod.torch as hvd

                if hvd.size():
                    return hvd.size()
            except (ModuleNotFoundError, ValueError, ImportError):
                pass

            # Try smdataparallel
            # smdistributed.dataparallel should be invoked via `mpirun`.
            # It supports EC2 machines with 8 GPUs per machine.
            if check_smdataparallel_env():
                try:
                    import smdistributed.dataparallel.torch.distributed as smdataparallel

                    if smdataparallel.get_world_size():
                        return smdataparallel.get_world_size()
                except (ModuleNotFoundError, ValueError, ImportError):
                    pass
        # Return default
        return 1

    def _get_worker_name(self):
        """Check horovod, smdataparallel, and torch.distributed."""
        # Try torch.distributed
        # torch.distributed is empty on Mac on Torch <= 1.2
        if hasattr(dist, "is_initialized") and dist.is_initialized():
            return f"worker_{dist.get_rank()}"
        else:
            # Try horovod
            try:
                import horovod.torch as hvd

                if hvd.size():
                    return f"worker_{hvd.rank()}"
            except (ModuleNotFoundError, ValueError, ImportError):
                pass

            # Try smdataparallel
            # smdistributed.dataparallel should be invoked via `mpirun`.
            # It supports EC2 machines with 8 GPUs per machine.
            if check_smdataparallel_env():
                try:
                    import smdistributed.dataparallel.torch.distributed as smdataparallel

                    if smdataparallel.get_world_size():
                        return f"worker_{smdataparallel.get_rank()}"
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

    def _collect_torch_profiling_data_if_profiler_enabled(self):
        if self.autograd_profiler_enabled is False:
            return
        if is_pt_1_8():
            records = torch.autograd._disable_profiler_legacy()
        else:
            records = torch.autograd._disable_profiler()
        self.autograd_profiler_enabled = False
        if is_pt_1_7():
            function_events = torch.autograd.profiler.EventList(
                torch.autograd.profiler.parse_event_records(records), use_cuda=self.use_cuda
            )
        elif is_pt_1_8():
            function_events = torch.autograd.profiler.EventList(
                torch.autograd.profiler.parse_legacy_records(records), use_cuda=self.use_cuda
            )
        else:
            function_events = torch.autograd.profiler.EventList(
                torch.autograd.profiler.parse_cpu_trace(records), use_cuda=self.use_cuda
            )

        for index, event in enumerate(function_events):
            if is_pt_1_8():
                cpu_time = event.time_range.start + self.start_profiler_time_us
                duration = event.time_range.elapsed_us() / float(CONVERT_TO_MICROSECS)
            else:
                cpu_time = event.cpu_interval.start + self.start_profiler_time_us
                # event.cpu_interval.start is in microseconds
                duration = event.cpu_interval.elapsed_us() / float(CONVERT_TO_MICROSECS)
            # timestamp is expected in seconds for record_trace_events
            timestamp = cpu_time / float(CONVERT_TO_MICROSECS)
            self.record_trace_events(
                training_phase="cpu_functions",
                op_name=event.name,
                phase="X",
                timestamp=timestamp,
                duration=duration,
                tid=event.thread,
                step_num=self.step,
                device="cpu",
            )
            for k in event.kernels:
                self.record_trace_events(
                    training_phase="gpu_functions-dev:" + str(k.device),
                    op_name=k.name,
                    phase="X",
                    timestamp=(k.interval.start + self.start_profiler_time_us)
                    / float(
                        CONVERT_TO_MICROSECS
                    ),  # timestamp expected is in seconds for record_trace_events
                    duration=k.interval.elapsed_us() / float(CONVERT_TO_MICROSECS),
                    tid=k.device,
                    step_num=self.step,
                    event_name=event.name,
                    device=k.device,
                    start_cpu_thread=event.thread,
                    cpu_thread_start_time=cpu_time,
                )

    # This hook is invoked by trainer prior to running the forward pass.
    def forward_pre_hook(self, module, inputs):
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

        ## prepararing for step metrics
        # last operation can be forward( eval loop is running or multiple forward for example RNN can have multiple call to forward of module)
        # or last operation can be backward (train backward loop just finished and we are at forward again)

        # we will log all outstanding forward and backward events
        self.log_outstanding_timeline_metrics()

        self.step_event = self._TraceEventData(
            phase="Step:" + str(self.mode),
            op_name="Step:" + str(self.mode),
            start_time=time.time(),
            dur=0,  # end time of step_event will be updated every time a forward event or backward is called after this
            pid=os.getpid(),
            step_num=str(self.mode_steps[self.mode]),
        )
        self.parent_forward_event = self._TraceEventData(
            phase="Forward",
            op_name=module._module_name,
            start_time=time.time(),
            dur=0,  # end time of parent_forward_event will be updated every time a forward event is called after this
            pid=os.getpid(),
            step_num=str(self.mode_steps[self.mode]),
        )

        self.profiler_config_parser.load_config()

        # Disable python profiling if the python profiler is currently profiling.
        if python_profiler:
            python_profiler.stop_profiling(
                StepPhase.STEP_START,
                end_mode=mode_keys_to_python_profile_mode(self.mode),
                end_step=self.step,
            )
            python_profiler.stop_profiling(StepPhase.STEP_START, self.step)
            if self.profiler_config_parser.should_save_metrics(
                MetricsCategory.PYTHON_PROFILING, self.step
            ):
                python_profiler.start_profiling(
                    StepPhase.STEP_START,
                    start_mode=mode_keys_to_python_profile_mode(self.mode),
                    start_step=self.step,
                )

        if self.autograd_profiler_enabled:
            self._collect_torch_profiling_data_if_profiler_enabled()

        # should we re-enable profiling for this step?
        if (
            self.profiler_config_parser.should_save_metrics(
                MetricsCategory.DETAILED_PROFILING, self.step
            )
            and not self.autograd_profiler_enabled
        ):
            self.autograd_profiler_enabled = True
            if is_pt_1_5():
                torch.autograd._enable_profiler(torch.autograd.ProfilerConfig(self.profiler, False))
                self.start_profiler_time_us = time.time() * CONVERT_TO_MICROSECS
            elif is_pt_1_7():
                torch.autograd._enable_profiler(
                    torch.autograd.ProfilerConfig(self.profiler, False, False, False)
                )
                self.start_profiler_time_us = time.time() * CONVERT_TO_MICROSECS
            elif is_pt_1_8():
                torch.autograd._enable_profiler_legacy(
                    torch.autograd.ProfilerConfig(self.profiler, False, False, False, False)
                )
                self.start_profiler_time_us = time.time() * CONVERT_TO_MICROSECS
            elif is_pt_1_6():
                torch.autograd._enable_profiler(
                    torch.autograd.ProfilerConfig(self.profiler, False, False)
                )
                self.start_profiler_time_us = time.time() * CONVERT_TO_MICROSECS
            else:
                self.logger.warn(
                    f"The detailed profiling using autograd profiler is not supported for torch version "
                    f"{torch.__version__}"
                )
                self.autograd_profiler_enabled = False

        if self.is_smdataparallel_profiling:
            # Stop smdataparallel profiling at end step
            stop_smdataparallel_profiler(
                smdataparallel, self.profiler_config_parser.config.local_path
            )
        self.is_smdataparallel_profiling = False
        if self.profiler_config_parser.should_save_metrics(
            MetricsCategory.SMDATAPARALLEL_PROFILING, self.step
        ):
            start_smdataparallel_profiler(
                smdataparallel, self.profiler_config_parser.config.local_path
            )
            self.is_smdataparallel_profiling = True

        if self._get_collections_to_save_for_step():
            self._initialize_writers()
            self._log_params(module)

        if self.last_saved_step is not None and not self.exported_collections:
            self.export_collections()
            self.exported_collections = True

        self.first_forward_submodule_name = None

    def record_tensor_value(self, tensor_name: str, tensor_value: torch.Tensor) -> None:
        """Used for registering functional directly, such as F.mse_loss()."""
        assert isinstance(
            tensor_value, torch.Tensor
        ), f"tensor_value={tensor_value} must be torch.Tensor"

        self._write_outputs(tensor_name, tensor_value)

    # This hook is invoked by trainer after running the forward pass.
    def forward_hook(self, module, inputs, outputs):
        # if this is first forward we will use start time of parent as start time, and end time as now
        cur_time = time.time()
        child_start_time = cur_time
        if self.parent_forward_event is not None:
            self.parent_forward_event.update_end_time(cur_time)
            # if this is first forward we will use start time of parent as start time, and end time as now
            child_start_time = self.parent_forward_event.start_time
        if self.step_event:
            self.step_event.update_end_time(cur_time)

        if len(self.forward_modules_profile_stats) > 0:
            # this child start_time is approcximated as last child end time
            child_start_time = self.forward_modules_profile_stats[-1].end_time

        event = self._TraceEventData(
            phase="Forward-SubModuleInternal",
            op_name=module._module_name,
            start_time=child_start_time,
            dur=cur_time - child_start_time,
            pid=os.getpid(),
            step_num=str(self.mode_steps[self.mode]),
        )
        self.forward_modules_profile_stats.append(event)
        if len(self.forward_modules_profile_stats) == 1:
            self.first_forward_submodule_name = module._module_name
        if not self._get_collections_to_save_for_step():
            return

        module_name = module._module_name
        # This overwhelms the logs; turn back on if you really need it
        # logger.debug("Processing the global step {0} for module {1}".format(self.step, module_name))

        # Output input tensor
        self._write_inputs(module_name, inputs)

        # Output output tensors
        self._write_outputs(module_name, outputs)
        self._save_custom_tensors_post_step()
        self.last_saved_step = self.step

    def backward_hook(self, tname):
        # Helper function that has access to the parameter name via
        # the scope in which it's defined.
        def back(grad):
            if self._get_collections_to_save_for_step():
                if grad is not None:
                    # self.logger.debug(f"Processing the backward step " f"{self.step} for {tname}")
                    self._save_for_tensor(self.GRADIENT_PREFIX + tname, grad)
            self._save_custom_tensors_post_step()

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

    def fhook(self, module, inputs, outputs):
        # we would stop profiling and restart from this phase
        if python_profiler:
            python_profiler.stop_profiling(
                StepPhase.FORWARD_PASS_END,
                end_mode=mode_keys_to_python_profile_mode(self.mode),
                end_step=self.step,
            )
            if self.profiler_config_parser.should_save_metrics(
                MetricsCategory.PYTHON_PROFILING, self.step
            ):
                python_profiler.start_profiling(
                    StepPhase.FORWARD_PASS_END,
                    start_mode=mode_keys_to_python_profile_mode(self.mode),
                    start_step=self.step,
                )

    def bhook(self, module, grad_input, grad_output):
        now = time.time()
        backward_st_time = now
        if self.parent_forward_event is not None:
            # note that we are approximating backward start_time here
            backward_st_time = self.parent_forward_event.end_time
            # we will not log step event yet, we will log step event only during forward pre-step
            # as step is FW + Backward
            self.log_outstanding_forward_stats_and_reset(log_step_event=False)
            # if this is first backward hook call, we will create backward event with start_Ts as last_forward_end_ts
            if self.parent_backward_event is None:
                self.parent_backward_event = self._TraceEventData(
                    phase="Backward(post-forward)",
                    op_name=module._module_name,
                    start_time=backward_st_time,
                    dur=now - backward_st_time,
                    pid=os.getpid(),
                    step_num=str(self.mode_steps[self.mode]),
                )
        if self.parent_backward_event:
            self.parent_backward_event.update_end_time(now)
        if self.step_event:
            self.step_event.update_end_time(now)
        # if this is not first backward we will use start time of parent as start time, and end time as now
        if len(self.backward_modules_profile_stats) > 0:
            # this child start_time is approcximated as last child end time
            child_start_time = self.backward_modules_profile_stats[-1].end_time
        else:
            child_start_time = backward_st_time
        event = self._TraceEventData(
            phase="Backward-SubModuleInternal",
            op_name=module._module_name,
            start_time=child_start_time,
            dur=now - child_start_time,
            pid=os.getpid(),
            step_num=str(self.mode_steps[self.mode]),
        )
        self.backward_modules_profile_stats.append(event)

    def _closure_for_registering_backward_hook(self, module):
        module.register_backward_hook(self.bhook)

    def count_parameters(self, model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            self.logger.info(f"name:{name} count_params:{param}")
            total_params += param
        self.logger.info(f"Total Trainable Params: {total_params}")
        return total_params

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
        # in case GPU is available but model has been loaded on CPU
        for parameter in module.parameters():
            self.profiler = (
                torch.autograd.ProfilerState.CUDA
                if parameter.is_cuda
                else torch.autograd.ProfilerState.CPU
            )
            self.use_cuda = parameter.is_cuda
            break
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
        module.register_forward_hook(self.fhook)

        # Set `self.forward_hook` as a callback for each submodule/layer.
        # `module.apply(fn)` calls fn for each submodule in module.children()
        module.apply(self._closure_for_registering_forward_hook)

        # Capture the gradient for each parameter in the net
        self._backward_apply(module)

        # TODO: Registering the backward hook causes issues in certain cases. There is a ‘Warning’ (
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_backward_hook#torch.nn.Module.register_backward_hook) for using this hook in certain cases.
        # The ‘__call_impl” in PyTorch Module class makes some assumptions about ‘results’ returned from the forward pass of the module.   It can not operate correctly if ‘forward’ pass returns anything other than dictionary of torch.Tensors.   Some of the torchvision.transform classes returned ‘PIL’ image object and backward hook used to crash.
        # In some cases, we have seen the the training hangs. Hence currently the following functionality is
        # commented. We can revisit it after understanding the PyTorch's implementation of backward hook.

        # module.apply(self._closure_for_registering_backward_hook)

        self.has_registered_module = True
        self.count_parameters(module)

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

    def close(self):
        self._cleanup()
        if python_profiler:
            python_profiler.start_profiling(
                StepPhase.STEP_END,
                start_mode=mode_keys_to_python_profile_mode(self.mode),
                start_step=self.mode_steps[self.mode],
            )

    def _cleanup(self):
        super()._cleanup()

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        return get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs)

    @staticmethod
    def _make_numpy_array(tensor_value):
        if isinstance(tensor_value, torch.Tensor):
            return tensor_value.to(torch.device("cpu")).data.numpy()
        return make_numpy_array(tensor_value)

    def should_save_dataloader_metrics(self, metrics_name):
        """Determine whether dataloader metrics for the provided metrics_name should be saved. We check for the next
        step since the dataloader metrics for the next step are collected on the current step.
        """
        return self.profiler_config_parser.should_save_metrics(
            MetricsCategory.DATALOADER_PROFILING, self.step + 1, metrics_name=metrics_name
        )
