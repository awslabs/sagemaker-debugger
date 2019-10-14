"""
Tests core functionality of naming workers when there are multiple processes.
See https://pytorch.org/tutorials/intermediate/ddp_tutorial.html to decide
how we want to support DistributedDataParallel with limited user configuration.

The key methods are
    torch.distributed.get_rank() - when manually spawning processes
"""
import numpy as nn
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.optim as optim
import shutil


import tornasole.pytorch as ts
from tornasole.trials import Trial, create_trial

out_dir = "/tmp/run"


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
    """Runs the training loop, no explicit Tornasole here."""
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


def run(rank, size, num_epochs=10, batch_size=128, num_batches=10):
    """Distributed function to be implemented later."""
    torch.manual_seed(1234)
    device = torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1)

    shutil.rmtree(out_dir, ignore_errors=True)
    hook = ts.TornasoleHook(
        out_dir=out_dir, save_config=ts.SaveConfig(save_steps=[0, 1, 5]), save_all=True
    )
    hook.register_hook(model)

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
        # print(f"Rank {dist.get_rank()}, epoch {epoch}: {epoch_loss / num_batches}")

    assert hook.get_worker_name() == f"worker_{dist.get_rank()}"
    # Race condition here where both workers attempt to move
    # /tmp/{out_dir}/END_OF_JOB.ts to {out_dir}/END_OF_JOB.ts
    try:
        hook._cleanup()
    except FileNotFoundError:
        pass


def average_gradients(model):
    """Gradient averaging."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def init_processes(rank, size, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def test_run_net_single_process():
    """Runs a single linear layer."""
    ts.reset_collections()
    device = torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    shutil.rmtree(out_dir, ignore_errors=True)
    hook = ts.TornasoleHook(
        out_dir=out_dir, save_config=ts.SaveConfig(save_steps=[0, 1, 5]), save_all=True
    )
    hook.register_hook(model)
    train(model=model, device=device, optimizer=optimizer)
    hook._cleanup()

    assert hook.get_worker_name() == "worker_0"

    trial = create_trial(path=out_dir)
    assert len(trial.workers()) == 1, f"trial.workers() = {trial.workers()}"
    assert len(trial.steps()) == 3, f"trial.steps() = {trial.steps()}"
    shutil.rmtree(out_dir, ignore_errors=True)


def test_run_net_distributed():
    """Runs a single linear layer on 2 processes."""
    # torch.distributed is empty on Mac on Torch <= 1.2
    if not hasattr(dist, "is_initialized"):
        return

    ts.reset_collections()
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # WARNING: assert statements do not cause test failure inside subprocesses
    # https://stackoverflow.com/questions/13400546/py-test-how-to-automatically-detect-an-exception-in-a-child-process
    assert all([not p.exitcode for p in processes]), f"Some processes failed. processes={processes}"

    out_dir = "/tmp/run"
    trial = create_trial(path=out_dir)
    assert len(trial.workers()) == 2, f"trial.workers() = {trial.workers()}"
    assert len(trial.steps()) == 3, f"trial.steps() = {trial.steps()}"
