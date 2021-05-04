# Standard Library
import logging
import os
from pathlib import Path

# Third Party
import pytest

# First Party
from smdebug.core.error_handling_agent import BASE_ERROR_MESSAGE
from smdebug.core.logger import DuplicateLogFilter, get_logger
from smdebug.core.utils import error_handling_agent


@pytest.fixture
def runtime_error_message(error_handling_message):
    return error_handling_message.format("RuntimeError")


@pytest.fixture
def value_error_message(error_handling_message):
    return error_handling_message.format("ValueError")


@pytest.fixture
def function_with_no_error_and_no_return_val():
    @error_handling_agent.catch_smdebug_errors()
    def do_nothing():
        pass

    return do_nothing


@pytest.fixture
def function_with_error_and_no_return_val(runtime_error_message):
    @error_handling_agent.catch_smdebug_errors()
    def raise_error():
        raise RuntimeError(runtime_error_message)

    return raise_error


@pytest.fixture
def function_with_no_error_and_return_val():
    @error_handling_agent.catch_smdebug_errors(default_return_val=False)
    def do_nothing():
        return True

    return do_nothing


@pytest.fixture
def function_with_error_and_return_val(value_error_message):
    @error_handling_agent.catch_smdebug_errors(default_return_val=False)
    def raise_error():
        raise ValueError(value_error_message)

    return raise_error


@pytest.fixture(autouse=True)
def set_up_logging_and_error_handling_agent(out_dir, stack_trace_filepath):
    """
    Set up each test to:
        - Add a logging handler to write all logs to a file (which will be used to verify caught errors in the tests)
        - Remove the duplicate logging filter
        - Reset the error handling agent after the test so that smdebug functionality is reenabled.
    """
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
    logger.removeHandler(file_handler)
    logger.addFilter(duplicate_log_filter)
    error_handling_agent.disable_smdebug = False


def test_no_smdebug_error(stack_trace_filepath, function_with_no_error_and_no_return_val):
    """
    Test that wrapping the error handling agent around a function that doesn't throw an error will allow the function
    to execute successfully without any errors thrown.
    """
    assert error_handling_agent.disable_smdebug is False
    return_val = function_with_no_error_and_no_return_val()
    assert return_val is None
    assert error_handling_agent.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""


def test_smdebug_error(
    stack_trace_filepath, function_with_error_and_no_return_val, runtime_error_message
):
    """
    Test that wrapping the error handling agent around a function that throws an error will allow the agent to
    catch the error and log the stack trace correctly.
    """
    assert error_handling_agent.disable_smdebug is False
    return_val = function_with_error_and_no_return_val()
    assert return_val is None
    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert runtime_error_message in stack_trace_logs


def test_no_smdebug_error_with_return_val(
    stack_trace_filepath, function_with_no_error_and_return_val
):
    """
    Test that wrapping the error handling agent around a function with a return value that doesn't throw an error will
    allow the function to execute successfully and return the correct value without any errors thrown.
    """
    assert error_handling_agent.disable_smdebug is False
    return_val = function_with_no_error_and_return_val()
    assert return_val is True
    assert error_handling_agent.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""


def test_smdebug_error_with_return_val(
    stack_trace_filepath, function_with_error_and_return_val, value_error_message
):
    """
    Test that wrapping the error handling agent around a function with a return value that throws an error will allow
    the agent to catch the error, log the stack trace correctly, and return the default value.
    """
    assert error_handling_agent.disable_smdebug is False
    return_val = function_with_error_and_return_val()
    assert return_val is False
    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert value_error_message in stack_trace_logs


def test_disabled_smdebug(
    stack_trace_filepath,
    function_with_error_and_no_return_val,
    function_with_no_error_and_return_val,
    function_with_error_and_return_val,
    runtime_error_message,
    value_error_message,
):
    """
    Test that disabling smdebug after an error is caught does the following:
        - If an smdebug function that doesn't throw an error is called, the default return value is returned without
          the function even being executed.
        - If an smdebug function that does return an error is called, the default return value is returned and no
          additional error is caught/logged.
    """
    assert error_handling_agent.disable_smdebug is False
    return_val = function_with_error_and_no_return_val()
    assert return_val is None
    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert runtime_error_message in stack_trace_logs

    os.remove(stack_trace_filepath)
    Path(stack_trace_filepath).touch()

    return_val = function_with_no_error_and_return_val()
    assert return_val is False
    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""

    return_val = function_with_error_and_return_val()
    assert return_val is False
    assert error_handling_agent.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""
