# Third Party
import pytest


@pytest.fixture
def stack_trace_filepath(out_dir):
    return f"{out_dir}/tmp.log"


@pytest.fixture
def error_handling_message():
    return (
        "If this {0} causes the test to fail, the error handling agent failed to catch the error!"
    )


@pytest.fixture
def custom_configuration_error_message():
    return "This error should have been thrown and not caught by the error handling agent!"
