# Standard Library
import os
import time
from datetime import datetime

# Third Party
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tests.core.utils import check_tf_events, delete_local_trials, verify_files
from torch.autograd import Variable

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.pytorch import Hook as PT_Hook

SMDEBUG_PT_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_pt/tests/"


def simple_pt_model(hook, steps=10, register_loss=False, with_timestamp=False):
    """
    Create a PT model. save_scalar() calls are inserted before, during and after training.
    Only the scalars with sm_metric=True will be written to a metrics file.
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.add_module("conv1", nn.Conv2d(1, 20, 5, 1))
            self.add_module("relu0", nn.ReLU())
            self.add_module("max_pool", nn.MaxPool2d(2, stride=2))
            self.add_module("conv2", nn.Conv2d(20, 50, 5, 1))
            self.add_module("relu1", nn.ReLU())
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

    model = Net().to(torch.device("cpu"))
    criterion = nn.NLLLoss()
    hook.register_module(model)
    if register_loss:
        hook.register_loss(criterion)

    scalars_to_be_saved = dict()
    ts = time.time()
    hook.save_scalar(
        "pt_num_steps", steps, sm_metric=True, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/pt_num_steps"] = (ts, steps)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ts = time.time()
    hook.save_scalar(
        "pt_before_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/pt_before_train"] = (ts, 1)

    hook.set_mode(ModeKeys.TRAIN)
    for i in range(steps):
        batch_size = 32
        data, target = torch.rand(batch_size, 1, 28, 28), torch.rand(batch_size).long()
        data, target = data.to(torch.device("cpu")), target.to(torch.device("cpu"))
        optimizer.zero_grad()
        output = model(Variable(data, requires_grad=True))
        if register_loss:
            loss = criterion(output, target)
        else:
            loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    ts = time.time()
    hook.save_scalar("pt_after_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None)
    scalars_to_be_saved["scalar/pt_after_train"] = (ts, 1)

    model.eval()
    hook.set_mode(ModeKeys.EVAL)
    with torch.no_grad():
        for i in range(steps):
            batch_size = 32
            data, target = torch.rand(batch_size, 1, 28, 28), torch.rand(batch_size).long()
            data, target = data.to("cpu"), target.to("cpu")
            output = model(data)
            if register_loss:
                loss = criterion(output, target)
            else:
                loss = F.nll_loss(output, target)
    return scalars_to_be_saved


def helper_pytorch_tests(collection, register_loss, save_config, with_timestamp):
    coll_name, coll_regex = collection

    run_id = "trial_" + coll_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(SMDEBUG_PT_HOOK_TESTS_DIR, run_id)

    hook = PT_Hook(
        out_dir=trial_dir,
        include_collections=[coll_name],
        save_config=save_config,
        export_tensorboard=True,
    )

    saved_scalars = simple_pt_model(
        hook, register_loss=register_loss, with_timestamp=with_timestamp
    )
    hook.close()
    verify_files(trial_dir, save_config, saved_scalars)
    if with_timestamp:
        check_tf_events(trial_dir, saved_scalars)


@pytest.mark.parametrize("collection", [("all", ".*"), ("scalars", "^scalar")])
@pytest.mark.parametrize(
    "save_config",
    [
        SaveConfig(save_steps=[0, 2, 4, 6, 8]),
        SaveConfig(
            {
                ModeKeys.TRAIN: SaveConfigMode(save_interval=2),
                ModeKeys.GLOBAL: SaveConfigMode(save_interval=3),
                ModeKeys.EVAL: SaveConfigMode(save_interval=1),
            }
        ),
    ],
)
@pytest.mark.parametrize("register_loss", [True, False])
@pytest.mark.parametrize("with_timestamp", [True, False])
def test_pytorch_save_scalar(collection, save_config, register_loss, with_timestamp):
    helper_pytorch_tests(collection, register_loss, save_config, with_timestamp)
    delete_local_trials([SMDEBUG_PT_HOOK_TESTS_DIR])
