from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tornasole import SaveConfig
from tornasole.pytorch.hook import *
from tornasole.pytorch.torch_collection import *
from tornasole.pytorch import reset_collections
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
import uuid
from tornasole.trials import create_trial
import shutil
import os


class Net(nn.Module):
    def __init__(self, mode='weights-bias-gradients', to_save=[]):
        super(Net, self).__init__()
        self.add_module('fc1', nn.Linear(20, 500))
        self.add_module('relu1', nn.ReLU())
        self.add_module('fc2', nn.Linear(500, 10))
        self.add_module('relu2', nn.ReLU())
        self.add_module('fc3', nn.Linear(10, 4))

        self.saved = dict()
        self.to_save = to_save
        self.step = -1
        self.mode = mode

        for name, param in self.named_parameters():
            pname = 'Net_' + name
            self.saved[pname] = dict()
            self.saved['gradient/' + pname] = dict()

        if self.mode == 'saveall':
            self.saved['fc1_input_0'] = dict()
            self.saved['relu1_input_0'] = dict()
            self.saved['fc2_input_0'] = dict()
            self.saved['relu2_input_0'] = dict()
            self.saved['fc3_input_0'] = dict()
            self.saved['Net_input_0'] = dict()
            self.saved['fc1_output0'] = dict()
            self.saved['relu1_output0'] = dict()
            self.saved['fc2_output0'] = dict()
            self.saved['relu2_output0'] = dict()
            self.saved['fc3_output0'] = dict()
            self.saved['Net_output0'] = dict()


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

        if self.mode == 'saveall':
            self.saved['fc1_input_0'][self.step] = x_in.data.numpy().copy()
            self.saved['relu1_input_0'][self.step] = fc1_out.data.numpy().copy()
            self.saved['fc2_input_0'][self.step] = relu1_out.data.numpy().copy()
            self.saved['relu2_input_0'][self.step] = fc2_out.data.numpy().copy()
            self.saved['fc3_input_0'][self.step] = relu2_out.data.numpy().copy()
            self.saved['Net_input_0'][self.step] = fc3_out.data.numpy().copy()

            self.saved['fc1_output0'][self.step] = fc1_out.data.numpy().copy()
            self.saved['relu1_output0'][self.step] = relu1_out.data.numpy().copy()
            self.saved['fc2_output0'][self.step] = fc2_out.data.numpy().copy()
            self.saved['relu2_output0'][self.step] = relu2_out.data.numpy().copy()
            self.saved['fc3_output0'][self.step] = fc3_out.data.numpy().copy()
            self.saved['Net_output0'][self.step] = out.data.numpy().copy()
        return out

# Create a tornasole hook. The initilization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initilization that enables logging of
# weights, biases and gradients in the model.
def create_tornasole_hook(output_dir, module=None, hook_type='saveall', save_steps=None):
    reset_collections()
    # Create a hook that logs weights, biases, gradients and inputs/ouputs of model
    if hook_type == 'saveall':
        hook = TornasoleHook(out_dir=output_dir,
                             save_config=SaveConfig(save_steps=save_steps),
                             save_all=True)
    elif hook_type == 'module-input-output':
        # The names of input and output tensors of a module are in following format
        # Inputs :  <module_name>_input_<input_index>, and
        # Output :  <module_name>_output
        # In order to log the inputs and output of a module, we will create a collection as follows:
        assert module is not None
        get_collection('l_mod').add_module_tensors(module, inputs=True, outputs=True)

        # Create a hook that logs weights, biases, gradients and inputs/outputs of model
        hook = TornasoleHook(out_dir=output_dir, save_config=SaveConfig(save_steps=save_steps),
                                include_collections=[CollectionKeys.WEIGHTS, CollectionKeys.GRADIENTS, CollectionKeys.BIASES, 'l_mod'])
    elif hook_type == 'weights-bias-gradients':
        save_config = SaveConfig(save_steps=save_steps)
        # Create a hook that logs ONLY weights, biases, and gradients
        hook = TornasoleHook(out_dir=output_dir, save_config=save_config)
    return hook

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

def helper_test_weights_bias_gradients(hook=None):
    prefix = str(uuid.uuid4())
    hook_type = 'weights-bias-gradients'
    device = torch.device("cpu")
    save_steps = [i * 20 for i in range(5)]
    model = Net(mode=hook_type, to_save=save_steps).to(device)
    json = hook is not None
    if not json:
        hook = create_tornasole_hook('test_output/test_hook_save_weightsbiasgradients/' \
                                     + prefix, model, hook_type, save_steps=save_steps)

    hook.register_hook(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, device, optimizer, num_steps=101, save_steps=save_steps)
    if not json:
        trial = create_trial(path='test_output/test_hook_save_weightsbiasgradients/' + prefix,
                        name='test output')
    else:
        trial = create_trial(path='test_output/test_hook_save_weightsbiasgradients/jsonloading',
                        name='test output')
    grads = ['gradient/Net_fc1.weight', 'gradient/Net_fc2.weight', 'gradient/Net_fc3.weight',
                'gradient/Net_fc1.bias', 'gradient/Net_fc2.bias', 'gradient/Net_fc3.bias']
    weights = ['Net_fc1.weight', 'Net_fc2.weight', 'Net_fc3.weight']
    bias = ['Net_fc1.bias', 'Net_fc2.bias', 'Net_fc3.bias']

    tensors = grads + bias + weights

    assert len(trial.available_steps()) == len(save_steps)
    for step in trial.available_steps():
        for tname in tensors:
            assert tname in trial.tensors()
            assert step in trial.tensor(tname).steps()
            saved_tensor = trial.tensor(tname).value(step)
            in_memory = model.saved[tname][step]
            assert np.allclose(in_memory, saved_tensor)
    if not json:
        addendum = prefix
    else:
        addendum = 'jsonloading'
    hook._cleanup()
    delete_local_trials(['test_output/test_hook_save_weightsbiasgradients/' + addendum])


def saveall_test_helper(hook=None):
    if hook is None:
        reset_collections()
    prefix = str(uuid.uuid4())
    hook_type = 'saveall'
    device = torch.device("cpu")
    save_steps = [i * 20 for i in range(5)]
    model = Net(mode=hook_type, to_save=save_steps).to(device)
    json = hook is not None
    if not json:
        hook = create_tornasole_hook('test_output/test_hook_saveall/' + prefix, model, hook_type, save_steps=save_steps)
    hook.register_hook(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, device, optimizer, num_steps=101, save_steps=save_steps)
    if not json:
        trial = create_trial(path='test_output/test_hook_saveall/' + prefix,
                         name='test output')
    else:
        trial = create_trial(path='test_output/test_hook_saveall/jsonloading',
                        name='test output')
    grads = ['gradient/Net_fc1.weight', 'gradient/Net_fc2.weight', 'gradient/Net_fc3.weight',
                'gradient/Net_fc1.bias', 'gradient/Net_fc2.bias', 'gradient/Net_fc3.bias']
    weights = ['Net_fc1.weight', 'Net_fc2.weight', 'Net_fc3.weight']
    bias = ['Net_fc1.bias', 'Net_fc2.bias', 'Net_fc3.bias']
    inputs = ['fc1_input_0', 'relu1_input_0', 'fc2_input_0', 'relu2_input_0', 'fc3_input_0']
    outputs = ['fc1_output0', 'relu1_output0', 'fc2_output0', 'relu2_output0', 'fc3_output0']
    tensors = grads + bias + weights + inputs + outputs

    assert len(trial.available_steps()) == len(save_steps)

    for step in trial.available_steps():
        for tname in tensors:
            assert tname in trial.tensors()
            assert step in trial.tensor(tname).steps()
            saved_tensor = trial.tensor(tname).value(step)
            in_memory = model.saved[tname][step]
            assert np.allclose(in_memory, saved_tensor)
    if not json:
        addendum = prefix
    else:
        addendum = 'jsonloading'
    hook._cleanup()
    delete_local_trials(['test_output/test_hook_saveall/' + addendum])


def helper_test_multi_collections(hook, out_dir):
    device = torch.device("cpu")
    hook_type = 'saveall'
    save_steps = [i for i in range(10)]
    model = Net(mode=hook_type, to_save=save_steps).to(device)
    hook.register_hook(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, device, optimizer, num_steps=101, save_steps=save_steps)
    trial = create_trial(path=out_dir,
                         name='test output')
    grads = ['gradient/Net_fc1.weight', 'gradient/Net_fc2.weight', 'gradient/Net_fc3.weight',
                'gradient/Net_fc1.bias', 'gradient/Net_fc2.bias', 'gradient/Net_fc3.bias']
    weights = ['Net_fc1.weight', 'Net_fc2.weight', 'Net_fc3.weight']
    bias = ['Net_fc1.bias', 'Net_fc2.bias', 'Net_fc3.bias']
    inputs = ['fc1_input_0', 'relu1_input_0', 'relu2_input_0']
    outputs = ['fc1_output0', 'relu1_output0', 'relu2_output0']
    tensors = grads + bias + weights + inputs + outputs

    assert len(trial.available_steps()) == len(save_steps)

    for tname in tensors:
        assert tname in trial.tensors()

def test_weightsbiasgradients_json():
    reset_collections()
    out_dir = 'test_output/test_hook_save_weightsbiasgradients/jsonloading'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/pytorch/test_json_configs/test_hook_weightsbiasgradients.json'
    hook = TornasoleHook.hook_from_config()
    helper_test_weights_bias_gradients(hook)

def test_weightsbiasgradients_call():
    helper_test_weights_bias_gradients()

def test_saveall_json():
    reset_collections()
    out_dir = 'test_output/test_hook_saveall/jsonloading'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/pytorch/test_json_configs/test_hook_saveall.json'
    hook = TornasoleHook.hook_from_config()
    saveall_test_helper(hook)

def test_saveall_params():
    saveall_test_helper()

# Test creating hook with multiple collections and save configs.
def test_multi_collection_json():
    reset_collections()
    out_dir = 'test_output/test_hook_multi_collection/jsonloading'
    shutil.rmtree(out_dir, True)
    os.environ[TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = 'tests/pytorch/test_json_configs/test_hook_multi_collections.json'
    hook = TornasoleHook.hook_from_config()
    helper_test_multi_collections(hook, out_dir)
