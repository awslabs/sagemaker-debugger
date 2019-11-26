# Third Party
import pytest

# First Party
from smdebug.trials import create_trial


@pytest.mark.slow  # 0:11 to run
def test_s3_read():
    path = "s3://tornasole-testing/dist-logs-10/"
    trial = create_trial(path)
    tensors = trial.tensor_names()
    assert len(tensors) == 17
    t = trial.tensor("gradients/dense_1/MatMul_grad/tuple/control_dependency_1:0")
    steps = t.steps()
    assert steps == [0]
    workers = t.workers(0)
    assert len(workers) == 16
    truth_table = t.value(0, worker="worker_10") == t.value(0, worker="worker_1")
    assert truth_table.all() == False
