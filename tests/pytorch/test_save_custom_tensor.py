# Standard Library
import shutil
from datetime import datetime

# Third Party
import torch
import torch.optim as optim

# First Party
from smdebug.core.collection import CollectionKeys
from smdebug.pytorch import SaveConfig
from smdebug.pytorch.hook import Hook as t_hook
from smdebug.trials import create_trial

# Local
from .utils import Net, train


def test_hook():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir = "/tmp/" + run_id
    hook = t_hook(
        out_dir=out_dir,
        save_config=SaveConfig(save_steps=[0, 1, 2, 3]),
        include_collections=["relu_activations"],
    )

    model = Net().to(torch.device("cpu"))
    hook.register_module(model)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, hook, torch.device("cpu"), optimizer, num_steps=10, save_custom_tensor=True)
    trial = create_trial(out_dir)
    custom_tensors = trial.tensor_names(collection=CollectionKeys.DEFAULT)
    assert len(custom_tensors) == 4
    shutil.rmtree(out_dir)
