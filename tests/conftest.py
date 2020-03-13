"""
Running all tests is as usual, `pytest tests`.
To only run the fast tests, run `pytest tests --skipslow`.
For CI, we will always run the full test suite.
"""

# Standard Library
import shutil

# Third Party
import pytest


def pytest_addoption(parser):
    # Anything taking longer than 2 seconds is slow
    # That's because we want to run the entire test suite in <1 minute for fast iteration
    # Running --skipslow is not comprehensive, but it will catch import errors and obvious bugs fast
    parser.addoption(
        "--skipslow", action="store_true", default=False, help="skip slow tests"
    )  # Anything taking longer than 2 seconds


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipslow"):
        # --skipslow given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="function")
def out_dir():
    """ Use this method to construct an out_dir.

    Then it will be automatically cleaned up for you, passed into the test method, and we'll have
    fewer folders lying around.
    """
    out_dir = "/tmp/test"
    shutil.rmtree(out_dir, ignore_errors=True)
    return out_dir


@pytest.fixture(scope="module", params=[True, False])
def tf_eager_mode(request):
    return request.param
