import pytest
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tornasole.pytorch as ts
from tornasole.trials import Trial, create_trial


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


@pytest.mark.slow  # 0:05 to run
def test_register_loss():
    """Test that the loss is saved as a tensor."""
    ts.reset_collections()
    out_dir = "/tmp/pytorch_test_loss"
    shutil.rmtree(out_dir, ignore_errors=True)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

    hook = ts.TornasoleHook(
        out_dir=out_dir,
        # With the default SaveConfig, the weights are not saved (only loss/gradient).
        # The weights tensors will be saved only at the final step, and only if they're a multiple
        # of save_interval. Issue with flushing?
        save_config=ts.SaveConfig(save_interval=1),
    )
    hook.register_hook(net)
    hook.register_loss(criterion)  # This is the important line

    batch_size = 1
    n_steps = 5
    # Use the same data at each step to test loss decreasing
    inputs, labels = torch.rand(batch_size, 3, 32, 32), torch.zeros(batch_size).long()
    for _ in range(n_steps):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # TODO(nieljare): Remove reliance on hook._cleanup()
    # What if the user has a training loop, then calls the Trials API in the same Python script
    # (like we do here). Then it'll crash, likewise in a Jupyter notebook.
    hook._cleanup()

    trial = create_trial(path=out_dir)
    loss_coll = hook.collection_manager.get("losses")
    assert len(loss_coll.get_tensor_names()) == 3

    loss_tensor = trial.tensor("CrossEntropyLoss_output_0")
    print(f"loss_tensor.steps() = {loss_tensor.steps()}")

    gradient_tensor = trial.tensor("gradient/Net_fc1.weight")
    print(f"gradient_tensor.steps() = {gradient_tensor.steps()}")

    weight_tensor = trial.tensor("Net_fc1.weight")
    print(f"weight_tensor.steps() = {weight_tensor.steps()}")

    assert len(trial.available_steps()) == n_steps
    assert len(weight_tensor.steps()) == n_steps
    assert len(gradient_tensor.steps()) == n_steps
    assert len(loss_tensor.steps()) == n_steps
    assert loss_tensor.value(0) > loss_tensor.value(4)
