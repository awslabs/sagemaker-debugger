# Using batch size 4 instead of 1024 decreases runtime from 35 secs to 4 secs.

# Standard Library
import os
import shutil
import time
from datetime import datetime

# Third Party
import mxnet as mx
import pytest
from mxnet import autograd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

# First Party
from smdebug import SaveConfig, modes
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.core.config_constants import CHECKPOINT_CONFIG_FILE_PATH_ENV_VAR
from smdebug.mxnet.hook import Hook as t_hook
from smdebug.trials import create_trial


def acc(output, label):
    return (output.argmax(axis=1) == label.astype("float32")).mean().asscalar()


def run_mnist(
    hook=None,
    set_modes=False,
    num_steps_train=None,
    num_steps_eval=None,
    epochs=2,
    save_interval=None,
    save_path="./saveParams",
):
    batch_size = 4
    normalize_mean = 0.13
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
    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(normalize_mean, 0.31)]
    )

    mnist_train = mnist_train.transform_first(transformer)
    mnist_valid = gluon.data.vision.FashionMNIST(train=False)

    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer), batch_size=batch_size, num_workers=4
    )

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

    if hook is not None:
        # Register the forward Hook
        hook.register_hook(net)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})
    hook.register_hook(softmax_cross_entropy)

    # Start the training.
    for epoch in range(epochs):
        train_loss, train_acc, valid_acc = 0.0, 0.0, 0.0
        tic = time.time()
        if set_modes:
            hook.set_mode(modes.TRAIN)

        i = 0
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
            i += 1
            if num_steps_train is not None and i >= num_steps_train:
                break
        # calculate validation accuracy
        if set_modes:
            hook.set_mode(modes.EVAL)
        i = 0
        for data, label in valid_data:
            data = data.as_in_context(mx.cpu(0))
            val_output = net(data)
            valid_acc += acc(val_output, label)
            loss = softmax_cross_entropy(val_output, label)
            i += 1
            if num_steps_eval is not None and i >= num_steps_eval:
                break
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
        if save_interval is not None and (epoch % save_interval) == 0:
            net.save_parameters("{0}/params_{1}.params".format(save_path, epoch))


@pytest.mark.slow  # 0:01 to run
def test_spot_hook():
    os.environ[
        CHECKPOINT_CONFIG_FILE_PATH_ENV_VAR
    ] = "./tests/mxnet/test_json_configs/checkpointconfig.json"
    checkpoint_path = "./savedParams"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    save_config = SaveConfig(save_steps=[10, 11, 12, 13, 14, 40, 50, 60, 70, 80])

    """
    Run the training for 2 epochs and save the parameter after every epoch.
    We expect that steps 0 to 14 will be written.
    """

    run_id_1 = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir_1 = "newlogsRunTest/" + run_id_1
    hook = t_hook(
        out_dir=out_dir_1, save_config=save_config, include_collections=["weights", "gradients"]
    )
    assert has_training_ended(out_dir_1) == False
    run_mnist(
        hook=hook,
        num_steps_train=10,
        num_steps_eval=10,
        epochs=2,
        save_interval=1,
        save_path=checkpoint_path,
    )

    """
    Run the training again for 4 epochs and save the parameter after every epoch.
    We DONOT expect that steps 0 to 14 are written.
    We expect to read steps 40, 50, 60, 70 and 80
    """
    run_id_2 = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir_2 = "newlogsRunTest/" + run_id_2
    hook = t_hook(
        out_dir=out_dir_2, save_config=save_config, include_collections=["weights", "gradients"]
    )
    assert has_training_ended(out_dir_2) == False
    run_mnist(
        hook=hook,
        num_steps_train=10,
        num_steps_eval=10,
        epochs=4,
        save_interval=1,
        save_path=checkpoint_path,
    )
    # Unset the environ variable before validation so that it won't affect the other scripts in py test environment.
    del os.environ[CHECKPOINT_CONFIG_FILE_PATH_ENV_VAR]

    # Validation
    print("Created the trial with out_dir {0} for the first training".format(out_dir_1))
    tr = create_trial(out_dir_1)
    assert tr
    available_steps_1 = tr.steps()
    assert 40 not in available_steps_1
    assert 80 not in available_steps_1
    print(available_steps_1)

    print("Created the trial with out_dir {0} for the second training".format(out_dir_2))
    tr = create_trial(out_dir_2)
    assert tr
    available_steps_2 = tr.steps()
    assert 40 in available_steps_2
    assert 50 in available_steps_2
    assert 60 in available_steps_2
    assert 70 in available_steps_2
    assert 80 in available_steps_2
    assert 0 not in available_steps_2
    assert 10 not in available_steps_2
    assert 11 not in available_steps_2
    assert 12 not in available_steps_2
    print(available_steps_2)

    print("Cleaning up.")
    shutil.rmtree(os.path.dirname(out_dir_1))
    shutil.rmtree(checkpoint_path)
