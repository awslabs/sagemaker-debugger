"""
Tests core functionality of naming workers when there are multiple processes.
See https://pytorch.org/tutorials/intermediate/ddp_tutorial.html to decide
how we want to support DistributedDataParallel with limited user configuration.

The key methods are
    torch.distributed.get_rank() - when manually spawning processes
"""
# Standard Library
import json
import os
import shutil
import time
from pathlib import Path

# Third Party
import numpy as nn
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import multiprocessing
from torch.multiprocessing import Process

# First Party
import smdebug.pytorch as smd
from smdebug.profiler.profiler_constants import DEFAULT_PREFIX
from smdebug.trials import create_trial


@pytest.fixture(autouse=True)
def set_up_smprofiler_config_path(monkeypatch):
    config_path = "tests/core/json_configs/simple_profiler_config_parser.json"
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)


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
    rank,
    size,
    include_workers="one",
    test_timeline=False,
    num_epochs=10,
    batch_size=128,
    num_batches=10,
):
    """Distributed function to be implemented later."""
    torch.manual_seed(1234)
    device = torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1)

    shutil.rmtree(out_dir, ignore_errors=True)

    hook = smd.Hook(
        out_dir=out_dir,
        save_config=smd.SaveConfig(save_steps=[0, 1, 5]),
        save_all=True,
        include_workers=include_workers,
    )

    hook.register_module(model)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        if test_timeline:
            hook.record_trace_events(
                training_phase="Training",
                op_name="TrainingEpochStart",
                phase="B",
                timestamp=start_time,
                rank=rank,
                epoch=epoch,
            )
        for _ in range(num_batches):
            optimizer.zero_grad()
            data, target = dataset(batch_size)
            output = model(data)
            loss = F.mse_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        end_time = time.time()
        if test_timeline:
            hook.record_trace_events(
                training_phase="Training",
                op_name="TrainingEpochEnd",
                phase="E",
                timestamp=end_time,
                rank=rank,
                duration=end_time - start_time,
                epoch=epoch,
            )
        # print(f"Rank {dist.get_rank()}, epoch {epoch}: {epoch_loss / num_batches}")

    assert hook._get_worker_name() == f"worker_{dist.get_rank()}"
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


def init_processes(rank, size, include_workers, test_timeline, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, include_workers, test_timeline)


def _run_net_distributed(include_workers="one", test_timeline=False):
    """Runs a single linear layer on 2 processes."""
    # torch.distributed is empty on Mac on Torch <= 1.2
    if not hasattr(dist, "is_initialized"):
        return
    multiprocessing.set_start_method("spawn", force=True)
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, include_workers, test_timeline, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # WARNING: assert statements do not cause test failure inside subprocesses
    # https://stackoverflow.com/questions/13400546/py-test-how-to-automatically-detect-an-exception-in-a-child-process
    assert all([not p.exitcode for p in processes]), f"Some processes failed. processes={processes}"

    out_dir = "/tmp/run"
    trial = create_trial(path=out_dir)
    return trial


@pytest.mark.slow  # 0:05 to run
def test_run_net_single_process():
    """Runs a single linear layer."""
    device = torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    shutil.rmtree(out_dir, ignore_errors=True)
    hook = smd.Hook(
        out_dir=out_dir, save_config=smd.SaveConfig(save_steps=[0, 1, 5]), save_all=True
    )
    hook.register_module(model)
    train(model=model, device=device, optimizer=optimizer)
    hook._cleanup()

    assert hook._get_worker_name() == "worker_0"

    trial = create_trial(path=out_dir)
    assert len(trial.workers()) == 1, f"trial.workers() = {trial.workers()}"
    assert len(trial.steps()) == 3, f"trial.steps() = {trial.steps()}"
    shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.slow  # 0:07 to run
def test_run_net_distributed_save_all_workers():
    trial = _run_net_distributed(include_workers="all")
    assert len(trial.workers()) == 2, f"trial.workers() = {trial.workers()}"
    assert len(trial.steps()) == 3, f"trial.steps() = {trial.steps()}"


@pytest.mark.slow  # 0:07 to run
def test_run_net_distributed_save_one_worker():
    trial = _run_net_distributed(include_workers="one")
    assert len(trial.workers()) == 1, f"trial.workers() = {trial.workers()}"
    assert len(trial.steps()) == 3, f"trial.steps() = {trial.steps()}"


@pytest.mark.slow
def test_run_net_distributed_save_all_test_timeline(monkeypatch):
    """
    This test checks if any of the timestamps recorded are negative
    """
    monkeypatch.setenv("ENV_BASE_FOLDER", out_dir)
    trial = _run_net_distributed(include_workers="all", test_timeline=True)
    assert len(trial.workers()) == 2, f"trial.workers() = {trial.workers()}"
    assert len(trial.steps()) == 3, f"trial.steps() = {trial.steps()}"

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    assert len(files) >= 2

    for file_name in files:
        with open(file_name) as timeline_file:
            events_dict = json.load(timeline_file)
            for e in events_dict:
                if e["name"].startswith("event"):
                    assert int(e["ts"]) >= 0
