# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# First Party
from smdebug import modes
from smdebug.pytorch import get_collection


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
        self.add_module("relu0", nn.ReLU())
        self.add_module("max_pool", nn.MaxPool2d(2, stride=2))
        self.add_module("conv2", nn.Conv2d(20, 50, 5, 1))
        relu_module = nn.ReLU()
        self.add_module("relu1", relu_module)
        get_collection("relu_activations").add_module_tensors(relu_module)
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


def train(model, hook, device, optimizer, num_steps=500, set_modes=False):
    if set_modes:
        hook.set_mode(modes.TRAIN)

    model.train()
    count = 0
    # for batch_idx, (data, target) in enumerate(train_loader):
    for i in range(num_steps):
        batch_size = 32
        data, target = torch.rand(batch_size, 1, 28, 28), torch.rand(batch_size).long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad=True))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, hook, device, num_steps=100, set_modes=False):
    if set_modes:
        hook.set_mode(modes.EVAL)

    for i in range(num_steps):
        batch_size = 32
        data, target = torch.rand(batch_size, 1, 28, 28), torch.rand(batch_size).long()
        data, target = data.to(device), target.to(device)
        output = model(Variable(data))
        loss = F.nll_loss(output, target)
