# MXNet

## Contents
- [Example](#mxnet-example)
- [Full API](#full-api)

## Support

- SageMaker Zero-Code-Change supported container: MXNet 1.6. See [AWS Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html) for more information.\
- Python API supported versions: MXNet 1.4, 1.5, 1.6.
- Only Gluon models are supported
- When the Gluon model is hybridized, inputs and outputs of intermediate layers can not be saved
- Parameter server based distributed training is not yet supported


## MXNet Example
```python
import smdebug.mxnet as smd
hook = smd.Hook(out_dir=args.out_dir)

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
# Here we register the block to smdebug
hook.register_block(net)


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
See the [API](api.md) page for details about Hook, Collection, SaveConfig, and ReductionConfig

See the [Analysis](analysis) page for details about analyzing a training job.
