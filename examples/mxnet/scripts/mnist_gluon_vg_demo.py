# Standard Library
import argparse
import random

# Third Party
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.gluon import nn

# First Party
from smdebug.mxnet import Hook, SaveConfig, modes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a mxnet gluon model for FashionMNIST dataset"
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default="s3://smdebug-testing/outputs/vg-demo",
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument(
        "--smdebug_path",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Reduce the number of training "
        "and evaluation steps to the give number if desired."
        "If this is not passed, trains for one epoch "
        "of training and validation data",
    )
    parser.add_argument("--save_frequency", type=int, default=100)
    opt = parser.parse_args()
    return opt


def test(ctx, net, val_data, num_steps=None):
    metric = mx.metric.Accuracy()
    for i, (data, label) in enumerate(val_data):
        if num_steps is not None and num_steps < i:
            break
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()


def train_model(
    net, epochs, ctx, learning_rate, momentum, train_data, val_data, hook, num_steps=None
):
    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(
        net.collect_params(), "sgd", {"learning_rate": learning_rate, "momentum": momentum}
    )
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        hook.set_mode(modes.TRAIN)
        for i, (data, label) in enumerate(train_data):
            if num_steps is not None and num_steps < i:
                break
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])
            # update metric at last.
            metric.update([label], [output])

            if i % 100 == 0 and i > 0:
                name, acc = metric.get()
                print("[Epoch %d Batch %d] Training: %s=%f" % (epoch, i, name, acc))

        name, acc = metric.get()
        print("[Epoch %d] Training: %s=%f" % (epoch, name, acc))

        hook.set_mode(modes.EVAL)
        name, val_acc = test(ctx, net, val_data, num_steps=num_steps)
        print("[Epoch %d] Validation: %s=%f" % (epoch, name, val_acc))


def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


def prepare_data():
    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("./data", train=True, transform=transformer),
        batch_size=100,
        shuffle=True,
        last_batch="discard",
    )

    val_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("./data", train=False, transform=transformer),
        batch_size=100,
        shuffle=False,
    )
    return train_data, val_data


# Create a model using gluon API. The hook is currently
# supports MXNet gluon models only.
def create_gluon_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(64, activation="relu"))
        net.add(nn.Dense(10))
    return net


# Create a hook. The initialization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initialization that enables logging of
# weights, biases and gradients in the model.
def create_hook(output_uri, save_frequency):
    # With the following SaveConfig, we will save tensors with the save_interval 100.
    save_config = SaveConfig(save_interval=save_frequency)

    # Create a hook that logs weights, biases and gradients while training the model.
    hook = Hook(
        out_dir=output_uri,
        save_config=save_config,
        include_collections=["weights", "gradients", "biases"],
    )
    return hook


def main():
    opt = parse_args()

    # these random seeds are only intended for test purpose.
    # for now, 128,12,2 could promise no assert failure with running test
    # if you wish to change the number, notice that certain steps' tensor value may be capable of variation
    if opt.random_seed:
        mx.random.seed(128)
        random.seed(12)
        np.random.seed(2)

    # Create a Gluon Model.
    net = create_gluon_model()

    # Create a hook for logging the desired tensors.
    # The output_uri is a the URI where the tensors will be saved. It can be local or s3://bucket/prefix
    output_uri = opt.smdebug_path if opt.smdebug_path is not None else opt.output_uri
    hook = create_hook(output_uri, opt.save_frequency)

    # Register the hook to the top block.
    hook.register_hook(net)

    # Start the training.
    train_data, val_data = prepare_data()

    train_model(
        net=net,
        epochs=2,
        ctx=mx.cpu(),
        learning_rate=1,
        momentum=0.9,
        train_data=train_data,
        val_data=val_data,
        hook=hook,
        num_steps=opt.num_steps,
    )


if __name__ == "__main__":
    main()
