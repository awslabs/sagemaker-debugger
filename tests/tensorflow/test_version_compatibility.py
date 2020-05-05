# Third Party
import pytest
import tensorflow as tf
from packaging import version


@pytest.mark.parametrize("supported_version", ["1.15"])
def test_current_version(supported_version):
    current_version = version.parse(tf.__version__)
    supported_version = version.parse(supported_version)
    assert (current_version.major, current_version.minor) == (
        supported_version.major,
        supported_version.minor,
    )
