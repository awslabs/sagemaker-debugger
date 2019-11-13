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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pt_utils import Net, get_dataloaders

# First Party
import smdebug.pytorch as smd
from smdebug.core.utils import SagemakerSimulator


def test_pytorch(script_mode: bool):
    with SagemakerSimulator() as sim:
        trainloader, testloader = get_dataloaders()
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        if script_mode:
            hook = smd.Hook(out_dir=sim.out_dir)
            hook.register_hook(net)
            hook.register_loss(criterion)

        for epoch in range(1):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                if True:
                    loss = criterion(outputs, labels)
                else:
                    loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    break

        print("Finished Training")

        from smdebug.trials import create_trial

        trial = create_trial(path=sim.out_dir)
        print(f"trial.steps() = {trial.steps()}")
        print(f"trial.tensors() = {trial.tensors()}")

        print(f"collection_manager = {hook.collection_manager}")

        weights_tensors = hook.collection_manager.get("weights").tensor_names
        print(f"'weights' collection tensors = {weights_tensors}")
        assert len(weights_tensors) > 0

        gradients_tensors = hook.collection_manager.get("gradients").tensor_names
        print(f"'gradients' collection tensors = {gradients_tensors}")
        assert len(gradients_tensors) > 0

        losses_tensors = hook.collection_manager.get("losses").tensor_names
        print(f"'losses' collection tensors = {losses_tensors}")
        assert len(losses_tensors) > 0

        assert all(
            [name in trial.tensors() for name in hook.collection_manager.get("losses").tensor_names]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script-mode", help="Manually create hooks instead of relying on ZCC", action="store_true"
    )
    args = parser.parse_args()

    test_pytorch(script_mode=args.script_mode)
