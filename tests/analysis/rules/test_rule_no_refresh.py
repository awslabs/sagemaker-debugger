# Standard Library
import uuid

# Third Party
from tests.analysis.utils import generate_data

# First Party
from smdebug.analysis.utils import no_refresh
from smdebug.exceptions import StepNotYetAvailable
from smdebug.rules import Rule
from smdebug.trials import create_trial


def test_no_refresh_invocation():
    class TestRule(Rule):
        def __init__(self, base_trial):
            super().__init__(base_trial=base_trial, action_str="")

        def set_required_tensors(self, step):
            for t in self.base_trial.tensor_names():
                self.req_tensors.add(t, steps=[step])

        def invoke_at_step(self, step):
            for t in self.req_tensors.get():
                if step == 0:
                    assert t.value(step + 1) is not None
                elif step == 1:
                    try:
                        t.value(step + 1)
                        assert False
                    except StepNotYetAvailable:
                        pass

    run_id = str(uuid.uuid4())
    base_path = "ts_output/rule_no_refresh/"
    path = base_path + run_id

    num_tensors = 3

    generate_data(
        path=base_path,
        trial=run_id,
        num_tensors=num_tensors,
        step=0,
        tname_prefix="foo",
        worker="algo-1",
        shape=(3, 3, 3),
    )
    generate_data(
        path=base_path,
        trial=run_id,
        num_tensors=num_tensors,
        step=1,
        tname_prefix="foo",
        worker="algo-1",
        shape=(3, 3, 3),
    )

    tr = create_trial(path)
    r = TestRule(tr)
    r.invoke(0)
    r.invoke(1)

    generate_data(
        path=base_path,
        trial=run_id,
        num_tensors=num_tensors,
        step=2,
        tname_prefix="foo",
        worker="algo-1",
        shape=(3, 3, 3),
    )

    # will not see step2 data
    with no_refresh(tr):
        r.invoke_at_step(1)

    # below will refresh
    try:
        r.invoke(1)
        assert False
    except AssertionError:
        pass
