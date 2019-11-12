# Third Party
import pytest

# First Party
from smdebug.exceptions import TensorUnavailableForStep
from smdebug.trials import create_trial


@pytest.mark.slow  # 0:01 to run
def test_fetch_tensor_with_missing_event_files():
    path = "s3://tornasole-testing/event-files-missing"

    trial = create_trial(path)
    try:
        # Get value from an event file that is present
        trial.tensor("gradients/pow_grad/sub:0").value(0)
    except TensorUnavailableForStep:
        assert False

    try:
        # Get value from an event file that is absent
        trial.tensor("gradients/pow_grad/sub:0").value(9)
        assert False
    except TensorUnavailableForStep:
        pass
