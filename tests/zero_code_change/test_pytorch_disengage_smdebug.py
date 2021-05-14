# Third Party
# Standard Library
import os
from unittest.mock import patch

import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tests.zero_code_change.pt_utils import Net, get_dataloaders

# First Party
import smdebug.pytorch as smd
from smdebug.core.config_validator import reset_config_validator
from smdebug.core.utils import SagemakerSimulator


@pytest.fixture(autouse=True)
def cleanup():
    yield
    os.environ.pop("USE_SMDEBUG", None)
    os.environ.pop("SM_HPS", None)
    reset_config_validator()


@patch("smdebug.core.config_validator.is_framework_version_supported", return_value=False)
def test_pytorch_with_unsupported_version(use_loss_module=False):
    smd.del_hook()

    sim_class = SagemakerSimulator
    with sim_class() as sim:
        trainloader, testloader = get_dataloaders()
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
            loss.backward()
            optimizer.step()
            if i == 99:
                break

        print("Finished Training")

        hook = smd.get_hook()
        assert hook is None
