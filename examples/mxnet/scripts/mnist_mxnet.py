# Standard Library
import argparse
import random
import time

# Third Party
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

# First Party
from smdebug import SaveConfig, modes
from smdebug.mxnet import Hook


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a mxnet gluon model for FashonMNIST dataset"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--output-uri",
        type=str,
        default="/opt/ml/output/tensors/smdebug",
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--random_seed", type=bool, default=False)
    opt = parser.parse_args()
    return opt


def acc(output, label):
    return (output.argmax(axis=1) == label.astype("float32")).mean().asscalar()


def train_model(batch_size, net, train_data, valid_data, lr, hook):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr})
    # Start the training.
    for epoch in range(50):
        train_loss, train_acc, valid_acc = 0.0, 0.0, 0.0
        tic = time.time()
        hook.set_mode(modes.TRAIN)
        for data, label in train_data:
            data = data.as_in_context(mx.cpu(0))
            # forward + backward
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)
        # calculate validation accuracy
        hook.set_mode(modes.EVAL)
        for data, label in valid_data:
            data = data.as_in_context(mx.cpu(0))
            valid_acc += acc(net(data), label)
        print(
            "Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec"
            % (
                epoch,
                train_loss / len(train_data),
                train_acc / len(train_data),
                valid_acc / len(valid_data),
                time.time() - tic,
            )
        )


def prepare_data(batch_size):
    mnist_train = datasets.FashionMNIST(train=True)
    X, y = mnist_train[0]
    ("X shape: ", X.shape, "X dtype", X.dtype, "y:", y)
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    X, y = mnist_train[0:10]
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])
    mnist_train = mnist_train.transform_first(transformer)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    mnist_valid = gluon.data.vision.FashionMNIST(train=False)
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer), batch_size=batch_size, num_workers=4
    )
    return train_data, valid_data


# Create a model using gluon API. The hook is currently
# supports MXNet gluon models only.
def create_gluon_model():
    # Create Model in Gluon
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
    net.initialize(init=init.Xavier(), ctx=mx.cpu())
    return net


# Create a hook. The initialization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initialization that enables logging of
# weights, biases and gradients in the model.
def create_hook(output_uri):
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3
    # (indexing starts with 0).
    save_config = SaveConfig(save_interval=1)

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
    # for now, 128,12,2 could promise no assert failure with running tests
    # if you wish to change the number, notice that certain steps' tensor value may be capable of variation
    if opt.random_seed:
        mx.random.seed(128)
        random.seed(12)
        np.random.seed(2)

    # Create a Gluon Model.
    net = create_gluon_model()

    # Create a hook for logging the desired tensors.
    # The output_s3_uri is a the URI for the s3 bucket where the tensors will be saved.
    # The trial_id is used to store the tensors from different trials separately.
    output_uri = opt.output_uri
    hook = create_hook(output_uri)

    # Register the hook to the top block.
    hook.register_hook(net)

    # Start the training.
    batch_size = opt.batch_size
    train_data, valid_data = prepare_data(batch_size)

    train_model(batch_size, net, train_data, valid_data, opt.learning_rate, hook)


if __name__ == "__main__":
    main()
