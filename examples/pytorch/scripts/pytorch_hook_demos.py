# Credit to the official pytorch mnist example set https://github.com/pytorch/examples/blob/master/mnist/main.py for help with this

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tornasole.pytorch import *
import tornasole.pytorch as ts


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.add_module('conv1', nn.Conv2d(1, 20, 5, 1))
        self.add_module('relu0', nn.ReLU())
        self.add_module('max_pool', nn.MaxPool2d(2, stride=2))
        self.add_module('conv2', nn.Conv2d(20, 50, 5, 1))
        self.add_module('relu1', nn.ReLU())
        self.add_module('max_pool2', nn.MaxPool2d(2, stride=2))
        self.add_module('fc1', nn.Linear(4*4*50, 500))
        self.add_module('relu2', nn.ReLU())
        self.add_module('fc2', nn.Linear(500, 10))


    def forward(self, x):
        x = self.relu0(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu1(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4*4*50)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad = True))
        loss = F.nll_loss(output, target)
        loss.backward()
        count += 1
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



# Create a tornasole hook. The initilization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initilization that enables logging of
# weights, biases and gradients in the model.
def create_tornasole_hook(output_dir, module=None, hook_type='saveall'):
    # Create a hook that logs weights, biases, gradients and inputs/ouputs of model every 10 steps while training.
    if hook_type == 'saveall':
        hook = TornasoleHook(out_dir=output_dir, save_config=SaveConfig(save_steps=[i * 10 for i in range(20)]), save_all=True)
    elif hook_type == 'module-input-output':
        # The names of input and output tensors of a module are in following format
        # Inputs :  <module_name>_input_<input_index>, and
        # Output :  <module_name>_output
        # In order to log the inputs and output of a module, we will create a collection as follows:
        assert module is not None
        get_collection('l_mod').add_module_tensors(module, inputs=True, outputs=True)

        # Create a hook that logs weights, biases, gradients and inputs/outputs of model every 5 steps from steps 0-100 while training.
        hook = TornasoleHook(out_dir=output_dir, save_config=SaveConfig(save_steps=[i * 5 for i in range(20)]),
                                include_collections=['weights', 'gradients', 'bias','l_mod'])
    elif hook_type == 'weights-bias-gradients':
        save_config = SaveConfig(save_steps=[i * 5 for i in range(20)])
        # Create a hook that logs ONLY weights, biases, and gradients every 5 steps (from steps 0-100) while training the model.
        hook = TornasoleHook(out_dir=output_dir, save_config=save_config)
    return hook

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output-uri', type=str, help="output directory to save data in", default='./tornasole-testing/demo/')
    parser.add_argument('--hook-type', type=str, choices=['saveall', 'module-input-output', 'weights-bias-gradients'], default='weights-bias-gradients')
    parser.add_argument('--mode', action="store_true")
    parser.add_argument('--rule_type', choices=['vanishing_grad', 'exploding_tensor', 'none'], default='none')
    args = parser.parse_args()

    device = torch.device("cpu")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)

    model = Net().to(device)

    if args.rule_type == 'vanishing_grad':
        lr, momentum = 1.0, 0.9
    elif args.rule_type == 'exploding_tensor':
        lr, momentum = 1000000.0, 0.9
    else:
        lr, momentum = args.lr, args.momentum

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    hook = create_tornasole_hook(output_dir=args.output_uri, module=model, hook_type=args.hook_type)
    hook.register_hook(model)

    for epoch in range(1, args.epochs + 1):
        if args.mode:
            hook.set_mode(ts.modes.TRAIN)
        train(args, model, device, train_loader, optimizer, epoch)
        if args.mode:
            hook.set_mode(ts.modes.EVAL)
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()
