# Using batch size 4 instead of 1024 decreases runtime from 35 secs to 4 secs.

from mxnet import gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time
import mxnet as mx
from tornasole import modes
import numpy as np

def acc(output, label):
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()


def run_mnist_gluon_model(hook=None, hybridize=False, set_modes=False, register_to_loss_block=False,
                          num_steps_train=None, num_steps_eval=None, make_input_zero=False, normalize_mean=0.13,
                          normalize_std=0.31):
    batch_size = 4
    if make_input_zero:
        mnist_train = datasets.FashionMNIST(train=True,
                                            transform=lambda data, label: (data.astype(np.float32) * 0, label))
        normalize_mean=0.0
    else:
        mnist_train = datasets.FashionMNIST(train=True)

    X, y = mnist_train[0]
    ('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)

    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    X, y = mnist_train[0:10]
    transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, 0.31)])

    mnist_train = mnist_train.transform_first(transformer)
    mnist_valid = gluon.data.vision.FashionMNIST(train=False)

    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer),
        batch_size=batch_size, num_workers=4)

    # Create Model in Gluon
    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(120, activation="relu"),
            nn.Dense(84, activation="relu"),
            nn.Dense(10))
    net.initialize(init=init.Xavier(),ctx=mx.cpu())
    if hybridize:
        net.hybridize(())

    if hook is not None:
    # Register the forward Hook
        hook.register_hook(net)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    if register_to_loss_block:
        hook.register_hook(softmax_cross_entropy)

    if set_modes:
        train_loss_name = eval_loss_name = 'loss_scalar'
        train_acc_name = eval_acc_name = 'acc'
    else:
        train_loss_name = 'train_loss_scalar'
        eval_loss_name = 'eval_loss_scalar'
        train_acc_name = 'train_acc'
        eval_acc_name = 'loss_acc'

    # Start the training.
    for epoch in range(1):
        train_loss, train_acc, valid_acc = 0., 0., 0.
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
            # hook.save_scalar(train_loss_name, train_loss)
            # hook.save_scalar(train_acc_name, train_acc)
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
            # hook.save_tensor('eval_labels', label)
            # hook.save_scalar(eval_acc_name, valid_acc)
            # hook.save_scalar(eval_loss_name, loss)
            i += 1
            if num_steps_eval is not None and i >= num_steps_eval:
                break
        print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
                epoch, train_loss/len(train_data), train_acc/len(train_data),
                valid_acc/len(valid_data), time.time()-tic))

    # for tests we have to call cleanup ourselves as destructor won't be called now
     # hook._cleanup()