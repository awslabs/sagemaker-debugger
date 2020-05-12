# Third Party
# Standard Library
import os

import pytest

# First Party
from smdebug.core.config_constants import MISSING_EVENT_FILE_RETRY_LIMIT_KEY
from smdebug.exceptions import TensorUnavailableForStep
from smdebug.trials import create_trial

os.environ[MISSING_EVENT_FILE_RETRY_LIMIT_KEY] = "5"


@pytest.mark.slow  # 0:01 to run
def test_fetch_tensor_with_present_event_files():
    """
        events files present: [0, 18, 27, 36, ...., 190]
        index files present: [0, 9, 18, 27, 36, ...., 190, 199]

        end_of_job file : present

    """
    path = "s3://smdebug-testing/resources/event-files-missing"

    trial = create_trial(path)
    # Get value from an event file that is present
    trial.tensor("gradients/pow_grad/sub:0").value(0)


@pytest.mark.slow  # 0:01 to run
def test_fetch_tensor_with_missing_event_file_but_next_event_file_present():
    """
        events files present: [0, 18, 27, 36, ...., 190]
        index files present: [0, 9, 18, 27, 36, ...., 190, 199]

        end_of_job file : present

    """
    path = "s3://smdebug-testing/resources/event-files-missing"

    trial = create_trial(path)
    with pytest.raises(TensorUnavailableForStep):
        # Get value from an event file that is absent
        trial.tensor("gradients/pow_grad/sub:0").value(9)


@pytest.mark.slow  # 0:01 to run
def test_fetch_tensor_with_missing_event_file_but_next_event_file_absent():
    """
        events files present: [0, 18, 27, 36, ...., 190]
        index files present: [0, 9, 18, 27, 36, ...., 190, 199]

        end_of_job file : present

    """
    path = "s3://smdebug-testing/resources/event-files-missing"

    trial = create_trial(path)
    with pytest.raises(TensorUnavailableForStep):
        # Get value from an event file that is absent
        trial.tensor("gradients/pow_grad/sub:0").value(199)


@pytest.mark.slow  # 0:01 to run
def test_fetch_tensor_with_missing_event_file_but_next_event_file_present_incomplete_job():
    """
        events files present: [0, 18, 27, 36, ...., 190]
        index files present: [0, 9, 18, 27, 36, ...., 190, 199]

        end_of_job file : present

    """
    path = "s3://smdebug-testing/resources/event-files-missing-incomplete"

    trial = create_trial(path)
    with pytest.raises(TensorUnavailableForStep):
        # Get value from an event file that is absent
        trial.tensor("gradients/pow_grad/sub:0").value(9)


@pytest.mark.slow  # 0:01 to run
def test_fetch_tensor_with_missing_event_file_but_next_event_file_absent_incomplete_job():
    """
        events files present: [0, 18, 27, 36, ...., 190]
        index files present: [0, 9, 18, 27, 36, ...., 190, 199]

        end_of_job file : absent

    """
    path = "s3://smdebug-testing/resources/event-files-missing-incomplete"
    trial = create_trial(path)
    with pytest.raises(TensorUnavailableForStep):
        # Get value from an event file that is absent
        trial.tensor("gradients/pow_grad/sub:0").value(199)
