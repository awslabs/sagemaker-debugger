# Standard Library
import argparse
import random

# Third Party
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.gluon import nn

# First Party
from smdebug.core.utils import SagemakerSimulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a mxnet gluon model for FashonMNIST dataset"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument(
        "--context", type=str, default="cpu", help="Context can be either cpu or gpu"
    )
    parser.add_argument(
        "--validate", type=bool, default=True, help="Run validation if running with smdebug"
    )

    opt = parser.parse_args()
    return opt


def fn_test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for i, (data, label) in enumerate(val_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()


def train_model():
    # DEFAULT CONSTANTS
    batch_size = 256
    epochs = 1
    momentum = 0.9
    learning_rate = 0.1
    mx.random.seed(128)
    random.seed(12)
    np.random.seed(2)

    ctx = mx.cpu()
    # Create a Gluon Model.
    net = create_gluon_model()

    # Start the training.
    train_data, val_data = prepare_data(batch_size)

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
        for i, (data, label) in enumerate(train_data):
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
        name, val_acc = fn_test(ctx, net, val_data)
        print("[Epoch %d] Validation: %s=%f" % (epoch, name, val_acc))


def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


def prepare_data(batch_size):
    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("/tmp", train=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        last_batch="discard",
    )

    val_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("/tmp", train=False, transform=transformer),
        batch_size=batch_size,
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


def validate():
    try:
        from smdebug.trials import create_trial
        from smdebug.mxnet import get_hook

        hook = get_hook()
        # Check if the hook was executed with the default
        # hook configuration
        assert hook.has_default_hook_configuration()
        out_dir = hook.out_dir
        print("Created the trial with out_dir {0}".format(out_dir))
        tr = create_trial(out_dir)
        global_steps = tr.steps()
        print("Global steps: " + str(global_steps))

        loss_tensor_name = tr.tensor_names(regex="softmaxcrossentropyloss._output_.")[0]
        print("Obtained the loss tensor " + loss_tensor_name)
        assert loss_tensor_name == "softmaxcrossentropyloss0_output_0"

        mean_loss_tensor_value_first_step = tr.tensor(loss_tensor_name).reduction_value(
            step_num=global_steps[0], reduction_name="mean", abs=False
        )

        mean_loss_tensor_value_last_step = tr.tensor(loss_tensor_name).reduction_value(
            step_num=global_steps[-1], reduction_name="mean", abs=False
        )

        print("Mean validation loss first step = " + str(mean_loss_tensor_value_first_step))
        print("Mean validation loss last step = " + str(mean_loss_tensor_value_last_step))
        assert mean_loss_tensor_value_first_step >= mean_loss_tensor_value_last_step

    except ImportError:
        print("smdebug libraries do not exist. Skipped Validation.")

    print("Validation Complete")


def test_integration_mxnet():

    json_file_contents = """
        {
            "S3OutputPath": "s3://sagemaker-test",
            "LocalPath": "/tmp/mxnet_integ_test",
            "CollectionConfigurations": [
                {
                    "CollectionName": "losses",
                    "CollectionParameters": {
                        "save_interval": 100
                    }
                }
            ]
        }
        """
    with SagemakerSimulator(json_file_contents=json_file_contents) as _:
        train_model()
        validate()


if __name__ == "__main__":
    test_integration_mxnet()
