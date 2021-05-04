"""
WARNING: This must be run manually, with the custom TensorFlow fork installed.
Not used in CI/CD. May be useful for DLC testing.

We'll import a forked version of PyTorch, then run the MNIST tutorial at
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
This should work without changing anything from the tutorial.
Afterwards, we read from the directory and ensure that all the values are there.
"""
# Standard Library
import argparse

# Third Party
import pytest
import torch
from tests.zero_code_change.pt_utils import helper_torch_train

# First Party
import smdebug.pytorch as smd
from smdebug.core.utils import SagemakerSimulator, ScriptSimulator


@pytest.mark.skipif(
    torch.__version__ == "1.7.0",
    reason="Disabling the test temporarily until we root cause the version incompatibility",
)
@pytest.mark.parametrize("script_mode", [False])
@pytest.mark.parametrize("use_loss_module", [True, False])
def test_pytorch(script_mode, use_loss_module):
    smd.del_hook()

    sim_class = ScriptSimulator if script_mode else SagemakerSimulator
    with sim_class() as sim:
        helper_torch_train(sim=sim, script_mode=script_mode, use_loss_module=use_loss_module)

        print("Finished Training")

        hook = smd.get_hook()
        print(f"hook = {hook}")
        # Check if the hook was executed with the default
        # hook configuration
        assert hook.has_default_hook_configuration()

        from smdebug.trials import create_trial

        trial = create_trial(path=sim.out_dir)
        print(f"trial.steps() = {trial.steps()}")
        print(f"trial.tensor_names() = {trial.tensor_names()}")

        print(f"collection_manager = {hook.collection_manager}")

        losses_tensors = hook.collection_manager.get("losses").tensor_names
        print(f"'losses' collection tensor_names = {losses_tensors}")
        assert len(losses_tensors) > 0

        assert all(
            [
                name in trial.tensor_names()
                for name in hook.collection_manager.get("losses").tensor_names
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-mode", help="Manually create hooks instead of relying on ZCC", action="store_true"
    )
    args = parser.parse_args()
    use_script_mode = args.script_mode

    test_pytorch(script_mode=use_script_mode, use_loss_module=True)
    test_pytorch(script_mode=use_script_mode, use_loss_module=False)
