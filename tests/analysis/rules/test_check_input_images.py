# Standard Library
import argparse

# Third Party
import mxnet as mx
from mxnet import autograd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

# First Party
import smdebug.mxnet as smd
from smdebug.mxnet import Hook, SaveConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a mxnet gluon model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--output-s3-uri",
        type=str,
        default="s3://tornasole-testing/saveall-mxnet-hook",
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument(
        "--smdebug_path",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument("--random_seed", type=bool, default=True)
    parser.add_argument(
        "--flag",
        type=bool,
        default=True,
        help="Bool variable that indicates whether parameters will be intialized to zero",
    )
    opt = parser.parse_args()
    return opt


def create_gluon_model():
    net = nn.HybridSequential()
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10),
    )
    net.initialize(init=init.Uniform(1), ctx=mx.cpu())
    return net


def train_model(batch_size, net, train_data, lr):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr})
    for epoch in range(1):
        for data, label in train_data:
            data = data.as_in_context(mx.cpu(0))
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)


def create_hook(output_s3_uri):
    save_config = SaveConfig(save_interval=1)
    custom_collect = smd.get_collection("inputData")
    custom_collect.save_config = save_config
    custom_collect.include([".*hybridsequential0_input_0"])
    hook = Hook(out_dir=output_s3_uri, save_config=save_config, include_collections=["inputData"])
    return hook


def prepare_data(batch_size, flag):
    mnist_train = datasets.FashionMNIST(train=True)
    if flag:
        transformer = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.286, 0.352)]
        )
    else:
        transformer = transforms.Compose([transforms.ToTensor()])
    mnist_train = mnist_train.transform_first(transformer)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train_data


def main():
    opt = parse_args()
    net = create_gluon_model()
    output_s3_uri = opt.smdebug_path if opt.smdebug_path is not None else opt.output_s3_uri
    hook = create_hook(output_s3_uri)
    hook.register_hook(net)
    train_data = prepare_data(64, opt.flag)
    train_model(64, net, train_data, 0.1, hook)


if __name__ == "__main__":
    main()
