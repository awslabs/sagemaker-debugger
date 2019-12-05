# Standard Library
import uuid

# Third Party
import pytest
from tests.analysis.utils import generate_data

# First Party
from smdebug.exceptions import StepNotYetAvailable, StepUnavailable, TensorUnavailable
from smdebug.trials import create_trial


@pytest.mark.slow  # 0:38 to run
def test_refresh_tensors():
    trial_name = str(uuid.uuid4())
    path = f"s3://smdebug-testing/outputs/rules_refresh_tensors-{uuid.uuid4()}/"
    num_steps = 8
    num_tensors = 10
    for i in range(1, num_steps, 2):
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
        )
    tr = create_trial(path + trial_name)
    assert len(tr.steps()) == 4

    with pytest.raises(TensorUnavailable):
        tr.tensor("bar")

    assert tr.tensor("foo_1") is not None
    # available
    assert tr.tensor("foo_1").value(num_steps - 1) is not None
    # not saved
    with pytest.raises(StepUnavailable):
        tr.tensor("foo_1").value(num_steps - 2)

    for i in range(num_steps, num_steps * 2):
        if i % 2 == 0:
            continue
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
        )

    # refreshed
    assert tr.tensor("foo_1").value(num_steps + 1) is not None
    with pytest.raises(StepUnavailable):
        tr.tensor("foo_1").value(num_steps)

    with pytest.raises(StepNotYetAvailable):
        tr.tensor("foo_1").value(num_steps * 3)
