# PyTorch

Supported PyTorch versions: 1.2, 1.3.

## Contents
- [Module Loss Example](#module-loss-example)
- [Functional Loss Example](#functional-loss-example)
- [Full API](#full-api)

## Module Loss Example
```
import smdebug.pytorch as smd
hook = smd.Hook(out_dir=args.out_dir)

class Model(nn.Module)
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return F.relu(self.fc(x))

net = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Register the hook and the loss
hook.register_module(net)
hook.register_loss(criterion)

# Training loop as usual
for (inputs, labels) in trainloader:
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## Functional Loss Example
```
import smdebug.pytorch as smd
hook = smd.Hook(out_dir=args.out_dir)

class Model(nn.Module)
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return F.relu(self.fc(x))

net = Model()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Register the hook
hook.register_module(net)

# Training loop, recording the loss at each iteration
for (inputs, labels) in trainloader:
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = F.cross_entropy(outputs, labels)

    # Manually record the loss
    hook.record_tensor_value(tensor_name="loss", tensor_value=loss)

    loss.backward()
    optimizer.step()
```

## Full API
See the [Common API](https://link.com) page for details about Collection, SaveConfig, and ReductionConfig.\
See the [Analysis](https://link.com) page for details about analyzing a training job.

## Hook
```
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

```
register_module(
    self,
    module,
)
```
Adds callbacks to the module for recording tensors.

* `module` (torch.nn.Module): The module to use.


```
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
