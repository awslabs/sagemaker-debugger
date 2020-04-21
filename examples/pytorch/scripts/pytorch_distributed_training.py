# Standard Library
import argparse
import os

# Third Party
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import multiprocessing
from torch.multiprocessing import Process

# First Party
import smdebug.pytorch as smd


class Net(nn.Module):
    """Returns f(x) = sigmoid(w*x + b)"""

    def __init__(self):
        super().__init__()
        self.add_module("fc", nn.Linear(1, 1))

    def forward(self, x):
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


def dataset(batch_size=4):
    """Return a dataset of (data, target)."""
    data = torch.rand(batch_size, 1)
    target = F.sigmoid(2 * data + 1)
    return data, target


def train(model, device, optimizer, num_steps=10):
    """Runs the training loop."""
    model.train()
    for i in range(num_steps):
        batch_size = 4
        data = torch.rand(batch_size, 1)
        target = F.sigmoid(2 * data + 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()


def run(
    rank, size, out_dir, zcc, include_workers="one", num_epochs=10, batch_size=128, num_batches=10
):
    """Distributed function to be implemented later."""
    torch.manual_seed(1234)
    device = torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1)

    if zcc is False:
        hook = smd.Hook(
            out_dir=out_dir,
            save_config=smd.SaveConfig(save_steps=[0, 1, 5]),
            save_all=True,
            include_workers=include_workers,
        )

        hook.register_module(model)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for _ in range(num_batches):
            optimizer.zero_grad()
            data, target = dataset(batch_size)
            output = model(data)
            loss = F.mse_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()


def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def init_processes(rank, size, out_dir, include_workers, zcc, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, out_dir, zcc, include_workers)


def _run_net_distributed(out_dir, include_workers="one", zcc=False):
    """Runs a single linear layer on 2 processes."""
    # torch.distributed is empty on Mac on Torch <= 1.2
    if not hasattr(dist, "is_initialized"):
        return
    multiprocessing.set_start_method("spawn", force=True)
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, out_dir, include_workers, zcc, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/tmp/run")
    parser.add_argument("--include_workers", type=str, default="one")
    parser.add_argument("--zcc", type=str2bool, default=False)
    args = parser.parse_args()
    out_dir = args.out_dir
    include_workers = args.include_workers
    zcc = args.zcc
    _run_net_distributed(out_dir, include_workers, zcc)
