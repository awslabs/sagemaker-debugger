# PyTorch

Supported PyTorch versions: 1.2+.

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
hook.register_hook(net)
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
hook.register_hook(net)

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
See the [Common API](https://link.com) page for details about Collection, SaveConfig, and ReductionConfig.

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

`out_dir` (str): Where to write the recorded tensors and metadata\
`export_tensorboard` (bool): Whether to use TensorBoard logs\
`tensorboard_dir` (str): Where to save TensorBoard logs\
`dry_run` (bool): If true, don't write any files\
`reduction_config` (ReductionConfig object): See the Common API page.\
`save_config` (SaveConfig object): See the Common API page.\
`include_regex` (list[str]): List of additional regexes to save\
`include_collections` (list[str]): List of collections to save\
`save_all` (bool): Saves all tensors and collections. May be memory-intensive and slow.\
`include_workers` (str): Used for distributed training, can also be "all".


```
record_tensor_value(
    self,
    tensor_name,
    tensor_value,
)
```
Store a tensor which is outside a torch.nn.Module, for example a functional loss.

`tensor_name` (str): The tensor name, used to access the tensor value in analysis.\
`tensor_value` (torch.Tensor): The tensor itself.
