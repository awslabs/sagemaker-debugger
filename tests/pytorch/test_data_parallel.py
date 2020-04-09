# Standard Library
import shutil

# Third Party
import torch
import torch.optim as optim
from tests.pytorch.utils import Net, train
from torch.nn.parallel import DataParallel

# First Party
from smdebug.pytorch import Hook, SaveConfig
from smdebug.trials import create_trial

out_dir = "/tmp/run"


def test_data_parallel():
    shutil.rmtree(out_dir, ignore_errors=True)

    hook = Hook(
        out_dir=out_dir,
        save_config=SaveConfig(save_steps=[0, 1, 5]),
        save_all=True,
        include_workers="one",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    if device == "cuda":
        model = DataParallel(model)

    hook.register_module(model)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, hook, torch.device(device), optimizer, num_steps=10)

    hook.close()

    trial = create_trial(out_dir)
    assert trial.steps() == [0, 1, 5]
    if device == "cpu":
        assert len(trial.tensor_names()) == 37
    else:
        assert len(trial.tensor_names()) > 37

    shutil.rmtree(out_dir, ignore_errors=True)
