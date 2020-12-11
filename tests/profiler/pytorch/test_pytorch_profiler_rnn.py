# Standard Library
import os
import shutil
import time

# Third Party
import pytest
import torch
import torch.nn as nn

# First Party
import smdebug.pytorch as smd
from smdebug.profiler import LocalAlgorithmMetricsReader
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.pytorch.utils import is_pt_1_5, is_pt_1_6, is_pt_1_7


class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output


def train_model(out_dir="/tmp/smdebug", training_steps=5):
    rnn = RNN(50, 20, 10)
    save_config = smd.SaveConfig(save_interval=500)
    hook = smd.Hook(out_dir=out_dir, save_all=True, save_config=save_config)

    loss_fn = nn.MSELoss()

    hook.register_module(rnn)
    hook.register_module(loss_fn)

    batch_size = 10
    TIMESTEPS = training_steps

    # Create some fake data
    batch = torch.randn(batch_size, 50)
    hidden = torch.zeros(batch_size, 20)
    target = torch.zeros(batch_size, 10)

    loss = 0
    for t in range(TIMESTEPS):
        hidden, output = rnn(batch, hidden)
        loss += loss_fn(output, target)
    loss.backward()
    hook.close()


@pytest.fixture()
def pytorch_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_pytorch_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


def test_pytorch_profiler_rnn(pytorch_profiler_config_parser, out_dir):
    train_model(out_dir)
    lt = LocalAlgorithmMetricsReader(out_dir)
    lt.refresh_event_file_list()
    events = lt.get_events(0, time.time() * 1000000)
    print(f"Number of events {len(events)}")
    if is_pt_1_5():
        assert len(events) <= 64
    elif is_pt_1_6() or is_pt_1_7():
        assert len(events) <= 85
    shutil.rmtree(out_dir, ignore_errors=True)
