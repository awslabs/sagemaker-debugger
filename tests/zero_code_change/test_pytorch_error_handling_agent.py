# Standard Library
import logging
import os

# Third Party
import pytest
from tests.zero_code_change.pt_utils import helper_torch_train

# First Party
from smdebug.core.error_handling_agent import BASE_ERROR_MESSAGE
from smdebug.core.logger import DuplicateLogFilter, get_logger
from smdebug.core.utils import error_handling_agent
from smdebug.pytorch import Hook
from smdebug.pytorch.collection import CollectionKeys
from smdebug.pytorch.singleton_utils import del_hook


@pytest.fixture
def torch_callback_error_message(error_handling_message):
    return error_handling_message.format("PyTorch callback error")


@pytest.fixture
def register_module_error_message(error_handling_message):
    return error_handling_message.format("register module error")


@pytest.fixture
def profiler_config_parser_error_message(error_handling_message):
    return error_handling_message.format("profiler config parser error")


@pytest.fixture
def hook_class_with_torch_callback_error(out_dir, torch_callback_error_message):
    class HookWithBadTorchCallback(Hook):
        """
        Hook subclass with error callback called directly from PyTorch at regular intervals in
        training.
        """

        def __init__(self, torch_error_message=torch_callback_error_message, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.torch_callback_error_message = torch_error_message

        @error_handling_agent.catch_smdebug_errors()
        def fhook(self, module, inputs, outputs):
            """
            Override the Hook's fhook callback to fail immediately.
            """
            raise RuntimeError(self.torch_callback_error_message)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadTorchCallback(out_dir=out_dir)

    return HookWithBadTorchCallback


@pytest.fixture
def hook_class_with_register_module_error(out_dir, register_module_error_message):
    class HookWithBadRegisterModule(Hook):
        """
        Hook subclass with faulty `register_module` function called directly from PyTorch
        """

        def __init__(
            self, torch_register_module_error_message=register_module_error_message, *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.register_module_error_message = torch_register_module_error_message

        @error_handling_agent.catch_smdebug_errors()
        def register_module(self, module):
            """
            Override the Hook's register_module function to fail immediately. This simulates a failure to register
            the PyTorch callbacks.
            """
            raise RuntimeError(self.register_module_error_message)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadRegisterModule(out_dir=out_dir)

    return HookWithBadRegisterModule


@pytest.fixture
def hook_class_with_torch_callback_and_register_module_error(
    out_dir, hook_class_with_torch_callback_error, hook_class_with_register_module_error
):
    class HookWithBadTorchCallbackAndRegisterModule(
        hook_class_with_torch_callback_error, hook_class_with_register_module_error
    ):
        """
        Hook subclass with error callbacks called from PyTorch.

        Due to the ordering, the first callback to error is the `register_module` callback.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadTorchCallbackAndRegisterModule(out_dir=out_dir)

    return HookWithBadTorchCallbackAndRegisterModule


@pytest.fixture
def hook_class_with_profiler_config_parser_error(out_dir, profiler_config_parser_error_message):
    class HookWithBadProfilerConfigParser(Hook):
        """
        Hook subclass with faulty `should_save_dataloader_metrics` function called directly from PyTorch
        """

        def __init__(
            self,
            torch_profiler_config_parser_error_message=profiler_config_parser_error_message,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.profiler_config_parser_error_message = torch_profiler_config_parser_error_message

        @error_handling_agent.catch_smdebug_errors(default_return_val=False)
        def should_save_dataloader_metrics(self, metrics_name):
            """
            Override the Hook's should_save_dataloader_metrics function to fail immediately. This simulates a failure
            in the profiler config parser in determining whether dataloader metrics should be saved for the current
            step.
            """
            raise RuntimeError(self.profiler_config_parser_error_message)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadProfilerConfigParser(out_dir=out_dir)

    return HookWithBadProfilerConfigParser


@pytest.fixture
def hook_class_with_torch_callback_error_and_custom_debugger_configuration(
    out_dir, custom_configuration_error_message, hook_class_with_torch_callback_error
):
    class HookWithBadTorchCallbackAndCustomDebuggerConfiguration(
        hook_class_with_torch_callback_error
    ):
        """
        Hook subclass that extends off of the PyTorch callback error subclass above to return a hook with a
        custom debugger configuration. Thus, any errors thrown should not be caught by the error handling agent.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(
                torch_error_message=custom_configuration_error_message, *args, **kwargs
            )

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadTorchCallbackAndCustomDebuggerConfiguration(
                out_dir=out_dir, include_collections=[CollectionKeys.ALL]
            )

    return HookWithBadTorchCallbackAndCustomDebuggerConfiguration


@pytest.fixture
def profiler_config_path(config_folder):
    return os.path.join(config_folder, "test_pytorch_profiler_config_parser.json")


@pytest.fixture
def hook_class_with_torch_callback_error_and_custom_profiler_configuration(
    out_dir,
    custom_configuration_error_message,
    monkeypatch,
    profiler_config_path,
    hook_class_with_torch_callback_error,
):
    class HookWithBadTorchCallbackAndCustomProfilerConfiguration(
        hook_class_with_torch_callback_error
    ):
        """
        Hook subclass that extends off of the PyTorch callback error subclass above to return a hook with a
        custom profiler configuration. Thus, any errors thrown should not be caught by the error handling agent.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(
                torch_error_message=custom_configuration_error_message, *args, **kwargs
            )

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            monkeypatch.setenv("SMPROFILER_CONFIG_PATH", profiler_config_path)
            return HookWithBadTorchCallbackAndCustomProfilerConfiguration(out_dir=out_dir)

    return HookWithBadTorchCallbackAndCustomProfilerConfiguration


@pytest.fixture(autouse=True)
def set_up_logging_and_error_handling_agent(out_dir, stack_trace_filepath):
    """
    Set up each test to:
        - Add a logging handler to write all logs to a file (which will be used to verify caught errors in the tests)
        - Remove the duplicate logging filter
        - Reset the error handling agent after the test so that smdebug is reenabled.
    """
    old_create_from_json = Hook.create_from_json_file
    del_hook()

    logger = get_logger()
    os.makedirs(out_dir)
    file_handler = logging.FileHandler(filename=stack_trace_filepath)
    logger.addHandler(file_handler)
    duplicate_log_filter = None
    for log_filter in logger.filters:
        if isinstance(log_filter, DuplicateLogFilter):
            duplicate_log_filter = log_filter
            break
    logger.removeFilter(duplicate_log_filter)

    yield

    Hook.create_from_json_file = old_create_from_json
    error_handling_agent.disable_smdebug = False
    error_handling_agent.hook = None

    logger.removeHandler(file_handler)
    logger.addFilter(duplicate_log_filter)


@pytest.mark.parametrize(
    "hook_type",
    [
        "torch_callback_error",
        "register_module_error",
        "profiler_config_parser_error",
        "torch_callback_and_register_module_error",
    ],
)
def test_pytorch_error_handling(
    hook_type,
    hook_class_with_torch_callback_error,
    hook_class_with_register_module_error,
    hook_class_with_profiler_config_parser_error,
    hook_class_with_torch_callback_and_register_module_error,
    out_dir,
    stack_trace_filepath,
    torch_callback_error_message,
    register_module_error_message,
    profiler_config_parser_error_message,
):
    """
    Test that an error thrown by an smdebug function is caught and logged correctly by the error handling agent. This
    test has four test cases:

    torch_callback_error: The PyTorch hook's `fhook` is overridden to always fail. The error handling agent should
    catch this error and log it once, and then disable smdebug for the rest of training.

    register_module_error: The PyTorch hook's `register_module` is overridden to always fail. The error handling
    agent should catch this error and log it once, and then disable smdebug for the rest of training.

    profiler_config_error: The PyTorch hook's `should_save_dataloader_metrics` is overridden to always fail. The error
    handler should catch this error and log it once, and then disable smdebug for the rest of training.

    torch_callback_and_register_module_error: Both the `fhook` and `register_module` functions are overridden to always
    fail. However, the `register_module` function is called first, so the error handling agent should only catch that
    error. Then because smdebug is disabled, the error raised by `fhook` should not be caught because the function
    shouldn't be even be called in the first place.

    Each hook needs to be initialized during its corresponding test, because the error handling agent is configured
    to a hook during the hook initialization.
    """
    assert error_handling_agent.disable_smdebug is False

    if hook_type == "torch_callback_error":
        hook_class = hook_class_with_torch_callback_error
        error_message = torch_callback_error_message
    elif hook_type == "register_module_error":
        hook_class = hook_class_with_register_module_error
        error_message = register_module_error_message
    elif hook_type == "profiler_config_parser_error":
        hook_class = hook_class_with_profiler_config_parser_error
        error_message = profiler_config_parser_error_message
    else:
        hook_class = hook_class_with_torch_callback_and_register_module_error
        error_message = (
            register_module_error_message
        )  # only `register_module` should error and get caught

    Hook.create_from_json_file = hook_class.create_from_json_file

    helper_torch_train()

    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()

        # check that one error was caught
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 1

        # check that the right error was caught (printed twice for each time caught)
        assert stack_trace_logs.count(error_message) == 2

        # `fhook` should not have errored if `register_module` already errored.
        if hook_type == "torch_callback_and_register_module_error":
            assert torch_callback_error_message not in stack_trace_logs


@pytest.mark.parametrize("custom_configuration", ["debugger", "profiler"])
def test_non_default_smdebug_configuration(
    custom_configuration,
    monkeypatch,
    profiler_config_path,
    out_dir,
    hook_class_with_torch_callback_error_and_custom_debugger_configuration,
    hook_class_with_torch_callback_error_and_custom_profiler_configuration,
    custom_configuration_error_message,
    stack_trace_filepath,
):
    """
    Test that the error handling agent does not catch errors when a custom smdebug configuration of smdebug is used.

    Each hook needs to be initialized during its corresponding test, because the error handling agent is configured
    to a hook during the hook initialization.
    """
    if custom_configuration == "debugger":
        hook_class = hook_class_with_torch_callback_error_and_custom_debugger_configuration
    else:
        hook_class = hook_class_with_torch_callback_error_and_custom_profiler_configuration

    Hook.create_from_json_file = hook_class.create_from_json_file

    # Verify the correct error gets thrown and doesnt get caught.
    with pytest.raises(RuntimeError, match=custom_configuration_error_message):
        helper_torch_train()
    assert error_handling_agent.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 0
        assert stack_trace_logs.count(custom_configuration_error_message) == 0
