# Standard Library
import uuid

# Third Party
import pytest
from tests.analysis.utils import generate_data

# First Party
from smdebug.analysis.utils import no_refresh
from smdebug.trials import create_trial


def help_test_refresh_with_range(path):
    trial_name = str(uuid.uuid4())
    num_steps = 8
    num_tensors = 10
    for i in range(num_steps):
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
        )
    tr = create_trial(path + trial_name, range_steps=(0, 5))
    assert len(tr.steps()) == 5
    for i in range(num_steps, num_steps * 2):
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
            export_colls=False,
        )
    assert len(tr.steps()) == 5


def help_test_refresh(path):
    trial_name = str(uuid.uuid4())
    num_steps = 8
    num_tensors = 10
    for i in range(num_steps):
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

    assert "foo_" + str(num_tensors + 1) not in tr.tensor_names()
    assert "foo_1" in tr.tensor_names()
    assert len(tr.steps()) == num_steps
    assert len(tr.tensor("foo_1").steps()) == num_steps

    for i in range(num_steps, num_steps * 2):
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
            export_colls=False,
        )
    assert len(tr.tensor("foo_1").steps()) == num_steps * 2
    assert len(tr.steps()) == num_steps * 2

    generate_data(
        path=path,
        trial=trial_name,
        num_tensors=num_tensors,
        step=num_steps * 2 + 1,
        tname_prefix="foo",
        worker="algo-1",
        shape=(3, 3, 3),
        export_colls=False,
    )
    assert len(tr.steps()) == num_steps * 2 + 1

    generate_data(
        path=path,
        trial=trial_name,
        num_tensors=num_tensors + 3,
        step=num_steps * 2 + 2,
        tname_prefix="foo",
        worker="algo-1",
        shape=(3, 3, 3),
        export_colls=False,
    )
    assert tr.tensor("foo_" + str(num_tensors + 1)) is not None


def help_test_no_refresh(path):
    trial_name = str(uuid.uuid4())
    num_steps = 8
    num_tensors = 10

    for i in range(num_steps):
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

    assert "foo_" + str(num_tensors + 1) not in tr.tensor_names()
    assert "foo_1" in tr.tensor_names()
    assert len(tr.steps()) == num_steps
    assert len(tr.tensor("foo_1").steps()) == num_steps

    for i in range(num_steps, num_steps * 2):
        generate_data(
            path=path,
            trial=trial_name,
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
            export_colls=False,
        )

    with no_refresh([tr]) as [tr]:
        assert len(tr.tensor("foo_1").steps()) == num_steps
        assert len(tr.steps()) == num_steps

    with no_refresh([tr]):
        assert len(tr.tensor("foo_1").steps()) == num_steps
        assert len(tr.steps()) == num_steps

    with no_refresh(tr):
        assert len(tr.tensor("foo_1").steps()) == num_steps
        assert len(tr.steps()) == num_steps


def test_no_refresh_local():
    help_test_no_refresh("ts_output/train/")


@pytest.mark.slow  # 0:37 to run
def test_no_refresh_s3():
    help_test_no_refresh(f"s3://smdebug-testing/outputs/rules-{uuid.uuid4()}/")


def test_refresh_with_range_local():
    help_test_refresh_with_range("ts_output/train/")


@pytest.mark.slow  # 0:36 to run
def test_refresh_with_range_s3():
    help_test_refresh_with_range(f"s3://smdebug-testing/outputs/rules-{uuid.uuid4()}/")


def test_refresh_local():
    help_test_refresh("ts_output/train/")


@pytest.mark.slow  # 0:47 to run
def test_refresh_s3():
    help_test_refresh(f"s3://smdebug-testing/outputs/rules-{uuid.uuid4()}/")
