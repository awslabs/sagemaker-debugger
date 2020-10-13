# Standard Library
import argparse
import json
import os
import time

# Third Party
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# First Party
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

transform_delay_in_seconds = 0.0
training_delay = 0.0


def transform_delay(img):
    time.sleep(transform_delay_in_seconds)
    return img


def train(args, net, device, hook):
    batch_size = args.batch_size
    epoch = args.epoch
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(transform_delay),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )

    loss_optim = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

    epoch_times = []

    if hook:
        hook.set_mode(modes.TRAIN)
    # train the model
    train_duration = time.time()
    for i in range(epoch):
        print("START TRAINING")
        start = time.time()
        net.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_optim(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        print("Epoch %d: train loss %.3f, in %.1f sec" % (i, train_loss, epoch_time))

    train_duration = time.time() - train_duration
    print(f"Total_Train_Duration={train_duration};")
    # calculate training time after all epoch
    p50 = np.percentile(epoch_times, 50)
    return p50


def main():
    global transform_delay_in_seconds
    global training_delay
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--transform_delay", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--training_delay", type=float, default=0.0)
    parser.add_argument("--disable_hook", type=bool, default=False)
    parser.add_argument("--write_profiler_config", type=bool, default=False)

    opt = parser.parse_args()

    opt.pin_memory = True if opt.pin_memory else False
    transform_delay_in_seconds = opt.transform_delay
    training_delay = opt.training_delay

    for key, value in vars(opt).items():
        print(f"{key}:{value}")
    # create model
    net = models.__dict__[opt.model](pretrained=True)
    if opt.gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device)

    if opt.write_profiler_config:
        profiler_dict = {}
        profiler_dict["ProfilingIntervalInMilliseconds"] = 500
        param_dict = {}
        param_dict["ProfilerEnabled"] = "True"
        param_dict["LocalPath"] = "/opt/ml/output/tensors/"
        profiler_dict["ProfilingParameters"] = param_dict
        with open("/home/profilerconfig.json", "w") as f:
            json.dump(profiler_dict, f)
        os.environ["SMPROFILER_CONFIG_PATH"] = "/home/profilerconfig.json"

    if not opt.disable_hook:
        print("Enabling the sagemaker debugger hook")
        hook = get_hook()
    else:
        print("Sagemaker Debugger hook is disabled.")
        hook = None

    # Start the training.
    median_time = train(opt, net, device, hook)
    print("Median training time per Epoch=%.1f sec" % median_time)


if __name__ == "__main__":
    main()
