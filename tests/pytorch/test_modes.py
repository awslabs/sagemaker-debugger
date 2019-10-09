from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tornasole import modes, SaveConfig, SaveConfigMode
from tornasole.pytorch.hook import *
from tornasole.pytorch.collection import *
from tornasole.pytorch import reset_collections
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
import uuid
from tornasole.trials import create_trial
import shutil
import os


class Net(nn.Module):
    def __init__(self, to_save=[]):
        super(Net, self).__init__()
        self.add_module('fc1', nn.Linear(20, 500))
        self.add_module('relu1', nn.ReLU())
        self.add_module('fc2', nn.Linear(500, 10))
        self.add_module('relu2', nn.ReLU())
        self.add_module('fc3', nn.Linear(10, 4))

        self.saved = dict()
        self.to_save = to_save
        self.step = -1

        for name, param in self.named_parameters():
            pname = 'Net_' + name
            self.saved[pname] = dict()
            self.saved['gradient/' + pname] = dict()

    def forward(self, x_in):
        self.step += 1

        for name, param in self.named_parameters():
            pname = 'Net_' + name
            self.saved[pname][self.step] = param.data.numpy().copy()

        fc1_out = self.fc1(x_in)
        relu1_out = self.relu1(fc1_out)
        fc2_out = self.fc2(relu1_out)
        relu2_out = self.relu2(fc2_out)
        fc3_out = self.fc3(relu2_out)
        out = F.log_softmax(fc3_out, dim=1)
        return out


def train(model, device, optimizer, num_steps=500, save_steps=[]):
    model.train()
    count = 0
    # for batch_idx, (data, target) in enumerate(train_loader):
    for i in range(num_steps):
        batch_size=32
        data, target = torch.rand(batch_size, 20), torch.rand(batch_size).long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad = True))
        loss = F.nll_loss(output, target)
        loss.backward()
        if i in save_steps:
            model.saved['gradient/Net_fc1.weight'][i] = model.fc1.weight.grad.data.numpy().copy()
            model.saved['gradient/Net_fc2.weight'][i] = model.fc2.weight.grad.data.numpy().copy()
            model.saved['gradient/Net_fc3.weight'][i] = model.fc3.weight.grad.data.numpy().copy()
            model.saved['gradient/Net_fc1.bias'][i] = model.fc1.bias.grad.data.numpy().copy()
            model.saved['gradient/Net_fc2.bias'][i] = model.fc2.bias.grad.data.numpy().copy()
            model.saved['gradient/Net_fc3.bias'][i] = model.fc3.bias.grad.data.numpy().copy()
        optimizer.step()

def delete_local_trials(local_trials):
    for trial in local_trials:
        shutil.rmtree(trial)

def helper_test_modes(hook=None, out_dir='./test_output/test_hook_modes/'):
    prefix = str(uuid.uuid4())
    device = torch.device("cpu")
    save_steps = [i for i in range(5)]
    model = Net(to_save=save_steps).to(device)
    json = hook is not None
    if hook is None:
        out_dir = out_dir + "/" + prefix
        hook = TornasoleHook(out_dir=out_dir, save_config=SaveConfig({modes.TRAIN: SaveConfigMode(save_steps=save_steps)}))

    hook.register_hook(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    hook.set_mode(mode=modes.TRAIN)
    train(model, device, optimizer, num_steps=10, save_steps=save_steps)
    trial = create_trial(path=out_dir, name='test output')

    assert len(trial.modes()) == 1
    assert len(trial.available_steps()) == 5
    assert len(trial.available_steps(mode=modes.TRAIN)) == 5
    assert len(trial.available_steps(mode=modes.EVAL)) == 0

    if hook is None:
        shutil.rmtree(out_dir)


def test_training_mode():
    reset_collections()
    helper_test_modes()


# Test creating hook with multiple collections and save configs.
def test_training_mode_json():
    reset_collections()
    out_dir = 'test_output/test_hook_modes/jsonloading'
    shutil.rmtree(out_dir, True)
    os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/pytorch/test_json_configs/test_modes.json'
    hook = TornasoleHook.hook_from_config()
    helper_test_modes(hook, out_dir)
    shutil.rmtree(out_dir, True)
