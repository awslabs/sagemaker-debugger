# Standard Library

# Third Party
import pytest

# First Party
import smdebug.tensorflow as smd
from smdebug.trials import create_trial

# Local
from .test_estimator_modes import help_test_mnist


@pytest.mark.slow  # 0:02 to run
def test_mnist_local(out_dir):
    help_test_mnist(out_dir, smd.SaveConfig(save_interval=2), num_train_steps=4, num_eval_steps=2)
    tr = create_trial(out_dir)
    assert len(tr.collection("losses").tensor_names) == 1
    for t in tr.collection("losses").tensor_names:
        assert len(tr.tensor(t).steps()) == 3
