# Standard Library

# Third Party
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# First Party
import smdebug.pytorch as smd
from smdebug.trials import create_trial


class Net(nn.Module):
    """CIFAR-10 classification network structure."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_net_and_train(out_dir, n_steps, use_loss_module=False, use_loss_functional=False):
    assert (
        use_loss_module != use_loss_functional
    ), "Exactly one of `use_loss_module` and `use_loss_functional` must be true."

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    hook = smd.Hook(out_dir=out_dir, save_config=smd.SaveConfig(save_interval=1))
    hook.register_module(net)
    if use_loss_module:
        hook.register_loss(criterion)

    batch_size = 1
    # Use the same data at each step to test loss decreasing
    inputs, labels = torch.rand(batch_size, 3, 32, 32), torch.zeros(batch_size).long()
    for _ in range(n_steps):
        optimizer.zero_grad()
        outputs = net(inputs)
        if use_loss_module:
            loss = criterion(outputs, labels)
        if use_loss_functional:
            loss = F.cross_entropy(outputs, labels)
            hook.record_tensor_value("nll_loss", tensor_value=loss)
        loss.backward()
        optimizer.step()

    # Users can call this method to immediately use the Trials API.
    hook.close()
    smd.del_hook()


@pytest.mark.slow  # 0:05 to run
def test_register_loss_functional(out_dir):
    """ Test that the loss (as F.cross_entropy_loss) is saved as a tensor. """
    n_steps = 5
    create_net_and_train(out_dir=out_dir, n_steps=n_steps, use_loss_functional=True)

    trial = create_trial(path=out_dir)
    loss_coll = trial.collection("losses")
    loss_tensor = trial.tensor("nll_loss_output_0")

    # Capture ['nll_loss_output_0']
    assert len(trial.tensor_names()) >= 1
    assert len(loss_coll.tensor_names) >= 1

    # Loss should be logged for all the steps since passed `available_steps = range(n_steps)`
    assert len(trial.steps()) == n_steps
    assert len(loss_tensor.steps()) == n_steps

    # Loss should be decreasing
    assert loss_tensor.value(0) > loss_tensor.value(4)


@pytest.mark.slow  # 0:05 to run
def test_register_loss_module(out_dir):
    """ Test that the loss (as nn.Module) is saved as a tensor.

    Also test that nothing else is saved under the default config.
    """
    n_steps = 5
    create_net_and_train(out_dir=out_dir, n_steps=n_steps, use_loss_module=True)

    trial = create_trial(path=out_dir)
    loss_coll = trial.collection("losses")
    loss_tensor = trial.tensor("CrossEntropyLoss_output_0")

    # Capture ['CrossEntropyLoss_output_0']
    assert len(trial.tensor_names()) == 1
    assert len(loss_coll.tensor_names) == 1

    # Loss should be logged for all the steps since passed `available_steps = range(n_steps)`
    assert len(trial.steps()) == n_steps
    assert len(loss_tensor.steps()) == n_steps

    # Loss should be decreasing
    assert loss_tensor.value(0) > loss_tensor.value(4)
