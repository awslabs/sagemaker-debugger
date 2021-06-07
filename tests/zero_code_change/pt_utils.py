# Standard Library
from typing import Tuple

# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from packaging import version

# First Party
import smdebug.pytorch as smd


def get_dataloaders() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Temporary Change to allow the test to run with pytorch 1.7 RC3
    # Smdebug breaks when num_workers>0 for Pytorch 1.7.0
    if version.parse(torch.__version__) >= version.parse("1.7.0"):
        num_workers = 0
    else:
        num_workers = 2

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=num_workers
    )

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    return trainloader, testloader


class Net(nn.Module):
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


def helper_torch_train(sim=None, script_mode=False, use_loss_module=False):
    trainloader, testloader = get_dataloaders()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if script_mode:
        hook = smd.Hook(out_dir=sim.out_dir)
        hook.register_module(net)
        hook.register_loss(criterion)

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        if use_loss_module:
            loss = criterion(outputs, labels)
        else:
            loss = F.cross_entropy(outputs, labels)
            if script_mode:
                hook.record_tensor_value(tensor_name="loss", tensor_value=loss)
        loss.backward()
        optimizer.step()

        if i == 499:
            break
