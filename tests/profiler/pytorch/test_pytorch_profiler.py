# Future
from __future__ import print_function

# Standard Library
import os
import time

# Third Party
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# First Party
from smdebug.profiler.algorithm_metrics_reader import LocalAlgorithmMetricsReader
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.pytorch import Hook, modes
from smdebug.pytorch.utils import is_pt_1_5, is_pt_1_6, is_pt_1_7


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
        self.add_module("relu0", nn.ReLU())
        self.add_module("max_pool", nn.MaxPool2d(2, stride=2))
        self.add_module("conv2", nn.Conv2d(20, 50, 5, 1))
        self.add_module("relu1", nn.ReLU())
        self.add_module("max_pool2", nn.MaxPool2d(2, stride=2))
        self.add_module("fc1", nn.Linear(4 * 4 * 50, 500))
        self.add_module("relu2", nn.ReLU())
        self.add_module("fc2", nn.Linear(500, 10))

    def forward(self, x):
        x = self.relu0(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu1(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, optimizer, hook):
    model.train()

    for i in range(10):
        batch_size = 32
        data, target = torch.rand(batch_size, 1, 28, 28), torch.rand(batch_size).long()
        hook.set_mode(modes.TRAIN)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad=True))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        hook.set_mode(modes.EVAL)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad=True))


@pytest.fixture()
def pytorch_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_pytorch_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


def test_pytorch_profiler(pytorch_profiler_config_parser, out_dir):
    device = torch.device("cpu")
    model = Net().to(device)
    hook = Hook(out_dir=out_dir)
    hook.register_hook(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    train(model, device, optimizer, hook)
    hook.close()
    lt = LocalAlgorithmMetricsReader(out_dir)
    lt.refresh_event_file_list()
    events = lt.get_events(0, time.time() * 1000000)
    print(f"Number of events {len(events)}")
    if is_pt_1_5():
        assert len(events) == 386
    elif is_pt_1_6():
        assert len(events) == 672
    elif is_pt_1_7():
        assert 220 <= len(events)
