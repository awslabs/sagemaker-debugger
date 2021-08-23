# MXNet

## Contents
- [Support](#support)
- [How to use](#how-to-use)
- [Example](#example)
- [Full API](#full-api)

---

## Support

- The official [AWS Deep Learning Container for MXNet](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#general-framework-containers) supports a zero script change experience, in which you need no modifications to your training script.
- This library supports the following versions when you use our API, which requires a few minimal changes to your training script: MXNet 1.4, 1.5, 1.6, and 1.7.
- This library only supports Gluon models.
- When the Gluon model is hybridized, inputs and outputs of intermediate layers can't be saved.
- This library does not yet support parameter server-based distributed training.

---

## How to use
### Using zero script change containers
In this case, you don't need to do anything to get the hook running. We encourage you to configure the hook from the SageMaker Python SDK so you can run different jobs with different configurations without having to modify your script. If you want access to the hook to configure certain things which can not be configured through the SageMaker SDK, you can retrieve the hook as follows.
```
import smdebug.mxnet as smd
hook = smd.Hook.create_from_json_file()
```
You can create the hook from `smdebug`'s python API as in the following section even in such containers.

### Bring your own container experience
#### 1. Create a hook
If you're using SageMaker, you configure the hook in SageMaker's Python SDK using the Estimator class. Instantiate it with
`smd.Hook.create_from_json_file()`. Otherwise, call the hook class constructor: `smd.Hook()`.

#### 2. Register the model to the hook
Call `hook.register_block(net)`.

#### 3. Take actions using the hook APIs

For a full list of actions that the hook APIs offer to construct hooks and save tensors, see [Common hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#common-hook-api) and [MXNet specific hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#mxnet-specific-hook-api).

---

## Example
```python
#######################################
# Creating a hook. See `API for Saving Tensors` page for more information.
import smdebug.mxnet as smd
hook = smd.Hook(out_dir=args.out_dir)
#######################################

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon import nn
net = nn.HybridSequential()
net.add(
    nn.Dense(128, activation='relu'),
    nn.Dense(64, activation='relu'),
    nn.Dense(10, activation="relu"),
)
net.initialize(init=init.Xavier(), ctx=mx.cpu())
softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr})


#######################################
# Register the block to smdebug
hook.register_block(net)
#######################################

batch_size = 100
mnist = mx.test_utils.get_mnist()
train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

for i in range(args.epochs):
    # Reset the train data iterator.
    train_data.reset()
    # Loop over the train data iterator.
    for batch in train_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        with ag.record():
            for x, y in zip(data, label):
                z = net(x)
                loss = softmax_cross_entropy_loss(z, y)
                loss.backward()
                outputs.append(z)
        metric.update(label, outputs)
        trainer.step(batch.data[0].shape[0])
    name, acc = metric.get()
    metric.reset()
```

---

## Full API
See the [API for Saving Tensors](api.md) page for details about hooks, collections, SaveConfig, and ReductionConfig.

See the [Analysis](analysis) page for details about analyzing a training job.
