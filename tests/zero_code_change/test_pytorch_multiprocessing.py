# Future
from __future__ import print_function

# Standard Library
import os
import shutil

# Third Party
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tests.zero_code_change.utils import build_json
from torchvision import datasets, transforms

# First Party
from smdebug.trials import create_trial

data_dir = "/tmp/pytorch-mnist-data"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(rank, model, device, dataloader_kwargs):
    # Training Settings
    batch_size = 64
    epochs = 1
    lr = 0.01
    momentum = 0.5

    torch.manual_seed(1 + rank)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        **dataloader_kwargs
    )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(1, epochs + 1):
        train_epoch(epoch, model, device, train_loader, optimizer)


def train_epoch(epoch, model, device, data_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx > 4:
            break
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()


def test_no_failure_with_torch_mp(out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)
    path = build_json(out_dir, save_all=True, save_interval="1")
    path = str(path)
    os.environ["SMDEBUG_CONFIG_FILE_PATH"] = path
    device = "cpu"
    dataloader_kwargs = {}
    cpu_count = 2 if mp.cpu_count() > 2 else mp.cpu_count()

    torch.manual_seed(1)

    model = Net().to(device)
    model.share_memory()  # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(cpu_count):
        p = mp.Process(target=train, args=(rank, model, device, dataloader_kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    trial = create_trial(out_dir)

    assert trial.num_workers == 1  # Ensure only one worker saved data
    assert len(trial.tensor_names()) > 20  # Ensure that data was saved
    assert trial.steps() == [0, 1, 2, 3]  # Ensure that steps were saved
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)
