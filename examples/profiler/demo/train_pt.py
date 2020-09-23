# Standard Library
import argparse
import time

# Third Party
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# First Party
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import Hook


def between_steps_bottleneck():
    time.sleep(1)


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
    "--epochs", default=2, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument("--enable_bottleneck", type=str2bool, default=True)
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
    _ = Hook(out_dir="")  # need this line so that import doesn't get removed by pre-commit

    start = time.time()
    # create model
    net = models.__dict__[args.arch](pretrained=True)

    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)
    batch_size = 256

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    validset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_valid
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print("Loaded training")

    # train the model
    for epoch in range(5):
        net.train()
        step = 0
        for _, (inputs, targets) in enumerate(trainloader):
            output = net(inputs)
            loss = loss_optim(output, targets)
            optimizer.zero_grad()
            loss.backward()
            if args.enable_bottleneck:
                if 9 < step < 21:
                    between_steps_bottleneck()
                optimizer.step()
            step += 1
    end = time.time()
    print("Time taken:", end - start)


if __name__ == "__main__":
    main()
