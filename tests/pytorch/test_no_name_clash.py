# Standard Library
import shutil
from tempfile import TemporaryDirectory

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tests.pytorch.utils import train

# First Party
import smdebug.pytorch as smd
from smdebug.trials import create_trial


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
        self.add_module("max_pool", nn.MaxPool2d(2, stride=2))
        self.add_module("conv2", nn.Conv2d(20, 50, 5, 1))
        self.relu = nn.ReLU()
        self.add_module("max_pool2", nn.MaxPool2d(2, stride=2))
        self.add_module("fc1", nn.Linear(4 * 4 * 50, 500))
        self.add_module("fc2", nn.Linear(500, 10))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_no_name_clash():
    out_dir = TemporaryDirectory().name

    hook = smd.Hook(
        out_dir=out_dir,
        save_config=smd.SaveConfig(save_steps=[0, 1, 5]),
        save_all=True,
        include_workers="one",
    )
    model = Net()
    hook.register_module(model)
    device = "cpu"
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, hook, torch.device(device), optimizer, num_steps=10)

    trial = create_trial(out_dir)
    assert trial.steps() == [0, 1, 5]

    assert len(trial.tensor_names(regex="relu.*")) == 6
    shutil.rmtree(out_dir, ignore_errors=True)
