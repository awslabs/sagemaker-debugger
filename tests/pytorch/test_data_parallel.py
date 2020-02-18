# Standard Library
import shutil

# Third Party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# First Party
import smdebug.pytorch as smd
from smdebug.trials import create_trial

out_dir = "/tmp/run"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        output = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_dataloader():
    data = torch.randn([100, 20], requires_grad=True)
    target = torch.from_numpy(np.random.randint(0, 9, (100)))
    data = data.float()
    dataset = TensorDataset(data, target)
    dataloader = DataLoader(dataset, batch_size=10)
    return dataloader


def test_data_parallel():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    model = model.to(device)
    if device == "cuda":
        model = DataParallel(model)
    epochs = 10
    optimizer = Adam(model.parameters(), lr=0.0001)

    dataloader = get_dataloader()

    shutil.rmtree(out_dir, ignore_errors=True)

    hook = smd.Hook(
        out_dir=out_dir,
        save_config=smd.SaveConfig(save_steps=[0, 1, 5]),
        save_all=True,
        include_workers="one",
    )

    hook.register_module(model)

    for i in range(epochs):
        model.train()

        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()

    trial = create_trial(out_dir)
    assert len(trial.tensor_names()) == 1
