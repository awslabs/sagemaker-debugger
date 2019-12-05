# MXNet

## Contents
- [Support](#support)
- [How to Use](#how-to-use)
- [Example](#mxnet-example)
- [Full API](#full-api)

---

## Support

### Versions
- Zero Script Change experience where you need no modifications to your training script is supported in the official [SageMaker Framework Container for MXNet 1.6](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html), or the [AWS Deep Learning Container for MXNet 1.6](https://aws.amazon.com/machine-learning/containers/).

- This library itself supports the following versions when you use our API which requires a few minimal changes to your training script: MXNet 1.4, 1.5, 1.6.

---

## How to Use
### Using Zero Script Change containers
In this case, you don't need to do anything to get the hook running. You are encouraged to configure the hook from the SageMaker python SDK so you can run different jobs with different configurations without having to modify your script. If you want access to the hook to configure certain things which can not be configured through the SageMaker SDK, you can retrieve the hook as follows.
```
import smdebug.mxnet as smd
hook = smd.Hook.create_from_json_file()
```
Note that you can create the hook from smdebug's python API as is being done in the next section even in such containers.

### Bring your own container experience
#### 1. Create a hook
If using SageMaker, you will configure the hook in SageMaker's python SDK using the Estimator class. Instantiate it with
`smd.Hook.create_from_json_file()`. Otherwise, call the hook class constructor, `smd.Hook()`.

#### 2. Register the model to the hook
Call `hook.register_block(net)`.

#### 3. (Optional) Configure Collections, SaveConfig and ReductionConfig
See the [Common API](api.md) page for details on how to do this.

---

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

## Full API
See the [Common API](https://link.com) page for details about Collection, SaveConfig, and ReductionConfig.\
See the [Analysis](https://link.com) page for details about analyzing a training job.

## Hook
```python
__init__(
    out_dir,
    export_tensorboard = False,
    tensorboard_dir = None,
    dry_run = False,
    reduction_config = None,
    save_config = None,
    include_regex = None,
    include_collections= None,
    save_all = False,
    include_workers = "one",
)
```
Initializes the hook. Pass this object as a callback to Keras' `model.fit(), model.evaluate(), model.evaluate()`.

* `out_dir` (str): Where to write the recorded tensors and metadata.
* `export_tensorboard` (bool): Whether to use TensorBoard logs.
* `tensorboard_dir` (str): Where to save TensorBoard logs.
* `dry_run` (bool): If true, don't write any files.
* `reduction_config` (ReductionConfig object): See the Common API page.
* `save_config` (SaveConfig object): See the Common API page.
* `include_regex` (list[str]): List of additional regexes to save.
* `include_collections` (list[str]): List of collections to save.
* `save_all` (bool): Saves all tensors and collections. May be memory-intensive and slow.
* `include_workers` (str): Used for distributed training, can also be "all".

```python
register_block(
    self,
    block,
)
```
Adds callbacks to the module for recording tensors.

* `block` (mx.gluon.Block): The block to use.

```python
save_scalar(
    self,
    name,
    value,
    searchable = False,
)
```
Call this method at any point in the training script to log a scalar value, such as accuracy.

* `name` (str): Name of the scalar. A prefix 'scalar/' will be added to it.
* `value` (float): Scalar value.
* `searchable` (bool): If True, the scalar value will be written to SageMaker Metrics.
