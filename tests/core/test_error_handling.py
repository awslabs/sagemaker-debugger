# Standard Library
import logging
import os
from pathlib import Path

# Third Party
import pytest

# First Party
from smdebug.core.error_handler import BASE_ERROR_MESSAGE
from smdebug.core.logger import get_logger
from smdebug.core.utils import error_handler


@pytest.fixture
def stack_trace_filepath(out_dir):
    return f"{out_dir}/tmp.log"


@pytest.fixture
def runtime_error_message():
    return (
        "If this RuntimeError causes the test to fail, the error handler failed to catch the error!"
    )


@pytest.fixture
def value_error_message():
    return (
        "If this ValueError causes the test to fail, the error handler failed to catch the error!"
    )


@pytest.fixture
def dummy_clean_function():
    @error_handler.catch_smdebug_errors()
    def do_nothing():
        pass

    return do_nothing


@pytest.fixture
def dummy_error_function(runtime_error_message):
    @error_handler.catch_smdebug_errors()
    def raise_error():
        raise RuntimeError(runtime_error_message)

    return raise_error


@pytest.fixture
def dummy_clean_function_with_return_val():
    @error_handler.catch_smdebug_errors(return_type=bool)
    def do_nothing():
        return True

    return do_nothing


@pytest.fixture
def dummy_error_function_with_return_val(value_error_message):
    @error_handler.catch_smdebug_errors(return_type=bool)
    def raise_error():
        raise ValueError(value_error_message)

    return raise_error


@pytest.fixture(autouse=True)
def set_up(out_dir, stack_trace_filepath):
    logger = get_logger()
    os.makedirs(out_dir)
    Path(stack_trace_filepath).touch()
    file_handler = logging.FileHandler(filename=stack_trace_filepath)
    logger.addHandler(file_handler)
    error_handler.disable_smdebug = False


def test_no_smdebug_error(stack_trace_filepath, dummy_clean_function):
    assert error_handler.disable_smdebug is False
    return_val = dummy_clean_function()
    assert return_val is None
    assert error_handler.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""


def test_smdebug_error(stack_trace_filepath, dummy_error_function, runtime_error_message):
    assert error_handler.disable_smdebug is False
    return_val = dummy_error_function()
    assert return_val is None
    assert error_handler.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert runtime_error_message in stack_trace_logs


def test_no_smdebug_error_with_return_val(
    stack_trace_filepath, dummy_clean_function_with_return_val
):
    assert error_handler.disable_smdebug is False
    return_val = dummy_clean_function_with_return_val()
    assert return_val is True
    assert error_handler.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""


def test_smdebug_error_with_return_val(
    stack_trace_filepath, dummy_error_function_with_return_val, value_error_message
):
    assert error_handler.disable_smdebug is False
    return_val = dummy_error_function_with_return_val()
    assert return_val is False
    assert error_handler.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert value_error_message in stack_trace_logs


def test_disabled_smdebug(
    stack_trace_filepath,
    dummy_error_function,
    dummy_clean_function_with_return_val,
    dummy_error_function_with_return_val,
    runtime_error_message,
    value_error_message,
):
    assert error_handler.disable_smdebug is False
    return_val = dummy_error_function()
    assert return_val is None
    assert error_handler.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert runtime_error_message in stack_trace_logs

    os.remove(stack_trace_filepath)
    Path(stack_trace_filepath).touch()

    return_val = dummy_clean_function_with_return_val()
    assert return_val is False
    assert error_handler.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""

    return_val = dummy_error_function_with_return_val()
    assert return_val is False
    assert error_handler.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        assert logs.read() == ""
