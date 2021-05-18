# Standard Library
import logging
import os

# Third Party
import pytest
from tests.zero_code_change.test_mxnet_gluon_integration import train_model

# First Party
from smdebug.core.error_handling_agent import BASE_ERROR_MESSAGE
from smdebug.core.logger import DuplicateLogFilter, get_logger
from smdebug.core.utils import error_handling_agent
from smdebug.mxnet import Hook
from smdebug.mxnet.collection import CollectionKeys
from smdebug.mxnet.singleton_utils import del_hook


@pytest.fixture
def mxnet_callback_error_message(error_handling_message):
    return error_handling_message.format("PyTorch callback error")


@pytest.fixture
def register_block_error_message(error_handling_message):
    return error_handling_message.format("register block error")


@pytest.fixture
def hook_class_with_mxnet_callback_error(out_dir, mxnet_callback_error_message):
    class HookWithBadMXNetCallback(Hook):
        """
        Hook subclass with error callback called directly from MXNet at regular intervals in
        training.
        """

        def __init__(self, mxnet_error_message=mxnet_callback_error_message, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mxnet_callback_error_message = mxnet_error_message

        @error_handling_agent.catch_smdebug_errors()
        def forward_hook(self, block, inputs, outputs):
            """
            Override the Hook's forward_hook callback to fail immediately.
            """
            raise RuntimeError(self.mxnet_callback_error_message)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadMXNetCallback(out_dir=out_dir)

    return HookWithBadMXNetCallback


@pytest.fixture
def hook_class_with_register_block_error(out_dir, register_block_error_message):
    class HookWithBadRegisterBlock(Hook):
        """
        Hook subclass with faulty `register_block` function called directly from PyTorch
        """

        def __init__(
            self, torch_register_block_error_message=register_block_error_message, *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.register_block_error_message = torch_register_block_error_message

        @error_handling_agent.catch_smdebug_errors()
        def register_block(self, block):
            """
            Override the Hook's register_block function to fail immediately. This simulates a failure to register
            the MXNet callbacks.
            """
            raise RuntimeError(self.register_block_error_message)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadRegisterBlock(out_dir=out_dir)

    return HookWithBadRegisterBlock


@pytest.fixture
def hook_class_with_mxnet_callback_and_register_block_error(
    out_dir, hook_class_with_mxnet_callback_error, hook_class_with_register_block_error
):
    class HookWithBadMXNetCallbackAndRegisterBlock(
        hook_class_with_mxnet_callback_error, hook_class_with_register_block_error
    ):
        """
        Hook subclass with error callbacks called from MXNet.

        Due to the ordering, the first callback to error is the `register_block` callback.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadMXNetCallbackAndRegisterBlock(out_dir=out_dir)

    return HookWithBadMXNetCallbackAndRegisterBlock


@pytest.fixture
def hook_class_with_mxnet_callback_error_and_custom_debugger_configuration(
    out_dir, custom_configuration_error_message, hook_class_with_mxnet_callback_error
):
    class HookWithBadMXNetCallbackAndCustomDebuggerConfiguration(
        hook_class_with_mxnet_callback_error
    ):
        """
        Hook subclass that extends off of the MXNet callback error subclass above to return a hook with a
        custom debugger configuration. Thus, any errors thrown should not be caught by the error handling agent.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(
                mxnet_error_message=custom_configuration_error_message, *args, **kwargs
            )

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadMXNetCallbackAndCustomDebuggerConfiguration(
                out_dir=out_dir, include_collections=[CollectionKeys.ALL]
            )

    return HookWithBadMXNetCallbackAndCustomDebuggerConfiguration


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
    ["mxnet_callback_error", "register_block_error", "mxnet_callback_and_register_block_error"],
)
def test_pytorch_error_handling(
    hook_type,
    hook_class_with_mxnet_callback_error,
    hook_class_with_register_block_error,
    hook_class_with_mxnet_callback_and_register_block_error,
    out_dir,
    stack_trace_filepath,
    mxnet_callback_error_message,
    register_block_error_message,
):
    """
    Test that an error thrown by an smdebug function is caught and logged correctly by the error handling agent. This
    test has three test cases:

    mxnedt_callback_error: The MXNet hook's `forward_hook` is overridden to always fail. The error handling agent should
    catch this error and log it once, and then disable smdebug for the rest of training.

    register_block_error: The MXNet hook's `register_block` is overridden to always fail. The error handling
    agent should catch this error and log it once, and then disable smdebug for the rest of training.

    mxnet_callback_and_register_block_error: Both the `forward_hook` and `register_block` functions are overridden to always
    fail. However, the `register_block` function is called first, so the error handling agent should only catch that
    error. Then because smdebug is disabled, the error raised by `forward_hook` should not be caught because the function
    shouldn't be even be called in the first place.

    Each hook needs to be initialized during its corresponding test, because the error handling agent is configured
    to a hook during the hook initialization.
    """
    assert error_handling_agent.disable_smdebug is False

    if hook_type == "mxnet_callback_error":
        hook_class = hook_class_with_mxnet_callback_error
        error_message = mxnet_callback_error_message
    elif hook_type == "register_block_error":
        hook_class = hook_class_with_register_block_error
        error_message = register_block_error_message
    else:
        hook_class = hook_class_with_mxnet_callback_and_register_block_error
        error_message = (
            register_block_error_message
        )  # only `register_block` should error and get caught

    Hook.create_from_json_file = hook_class.create_from_json_file

    train_model()

    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()

        # check that one error was caught
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 1

        # check that the right error was caught (printed twice for each time caught)
        assert stack_trace_logs.count(error_message) == 2

        # `forward_hook` should not have errored if `register_block` already errored.
        if hook_type == "mxnet_callback_and_register_block_error":
            assert mxnet_callback_error_message not in stack_trace_logs


def test_non_default_smdebug_configuration(
    out_dir,
    hook_class_with_mxnet_callback_error_and_custom_debugger_configuration,
    custom_configuration_error_message,
    stack_trace_filepath,
):
    """
    Test that the error handling agent does not catch errors when a custom smdebug configuration of smdebug is used.

    Each hook needs to be initialized during its corresponding test, because the error handling agent is configured
    to a hook during the hook initialization.
    """
    Hook.create_from_json_file = (
        hook_class_with_mxnet_callback_error_and_custom_debugger_configuration.create_from_json_file
    )

    # Verify the correct error gets thrown and doesnt get caught.
    with pytest.raises(RuntimeError, match=custom_configuration_error_message):
        train_model()
    assert error_handling_agent.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 0
        assert stack_trace_logs.count(custom_configuration_error_message) == 0
