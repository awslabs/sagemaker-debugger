# Standard Library
import argparse
import time

# Third Party
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

# First Party
from smdebug.pytorch import Hook, SaveConfig

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data_dir", default="~/.pytorch/datasets/imagenet", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
args = parser.parse_args()


def main():
    start = time.time()
    # create model
    net = models.__dict__[args.arch](pretrained=True)
    device = torch.device("cpu")
    net.to(device)
    # register the hook

    hook = create_hook("./output_resnet", net, save_interval=50)

    hook.register_hook(net)
    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

    print("Loaded training")
    batch_size = 64
    # train the model
    for epoch in range(1):
        for i in range(4096):
            # Synthetic data generated here
            data_in = torch.rand(batch_size, 3, 64, 64)
            target = torch.zeros(batch_size).long()
            data_in, target = Variable(data_in), Variable(target)
            output = net(data_in)
            loss = loss_optim(output, target)
            if i % 10 == 0:
                print("Step", i, "Epoch", epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    end = time.time()
    print("Time taken:", end - start)


# Create a hook. The initilization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initilization that enables logging of
# weights, biases and gradients in the model.
def create_hook(output_dir, module, trial_id="trial-resnet", save_interval=100):
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3
    # (indexing starts with 0) and then continue to save tensors at interval of
    # 100,000 steps. Note: union operation is applied to produce resulting config
    # of save_steps and save_interval params.
    save_config = SaveConfig(save_interval)

    # The names of input and output tensors of a block are in following format
    # Inputs :  <block_name>_input_<input_index>, and
    # Output :  <block_name>_output
    # In order to log the inputs and output of a model, we will create a collection as follows

    # Create a hook that logs weights, biases, gradients of model while training.
    hook = Hook(out_dir=output_dir)
    return hook


if __name__ == "__main__":
    main()
