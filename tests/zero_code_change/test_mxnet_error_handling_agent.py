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
    return error_handling_message.format("MXNet callback error")


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

    import mxnet

    global global_smdebug_hook
    global global_hook_initialized

    mxnet.gluon.block.global_smdebug_hook = None
    mxnet.gluon.block.global_hook_initialized = False

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


def test_non_default_smdebug_configuration(
    out_dir,
    hook_class_with_mxnet_callback_error_and_custom_debugger_configuration,
    custom_configuration_error_message,
    stack_trace_filepath,
):
    """
    Test that the error handling agent does not catch errors when a custom smdebug configuration of smdebug is used.

    The hook needs to be initialized during its corresponding test, because the error handling agent is configured
    to a hook during the hook initialization.
    """
    Hook.create_from_json_file = (
        hook_class_with_mxnet_callback_error_and_custom_debugger_configuration.create_from_json_file
    )

    # Verify the correct error gets thrown and doesnt get caught. MXNet handles errors thrown in `forward_hook` and
    # throws its own ValueError, so we look for that error here.
    with pytest.raises(ValueError, match=custom_configuration_error_message):
        train_model()

    assert error_handling_agent.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 0
        assert stack_trace_logs.count(custom_configuration_error_message) == 0


def test_mxnet_error_handling(
    hook_class_with_mxnet_callback_error,
    out_dir,
    stack_trace_filepath,
    mxnet_callback_error_message,
):
    """
    Test that an error thrown by an smdebug function is caught and logged correctly by the error handling agent. This
    test has one test case:

    mxnet_callback_error: The MXNet hook's `forward_hook` is overridden to always fail. The error handling agent should
    catch this error and log it once, and then disable smdebug for the rest of training.

    The hook needs to be initialized during its corresponding test, because the error handling agent is configured
    to a hook during the hook initialization.
    """
    assert error_handling_agent.disable_smdebug is False

    Hook.create_from_json_file = hook_class_with_mxnet_callback_error.create_from_json_file

    train_model()

    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()

        # check that one error was caught
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 1

        # check that the right error was caught (printed twice for each time caught)
        assert stack_trace_logs.count(mxnet_callback_error_message) == 2
