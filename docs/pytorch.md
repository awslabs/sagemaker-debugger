# PyTorch

## Contents
- [Support](#support)
- [How to Use](#how-to-use)
- [Module Loss Example](#module-loss-example)
- [Functional Loss Example](#functional-loss-example)
- [Full API](#full-api)

## Support

### Versions
- Zero Script Change experience where you need no modifications to your training script is supported in the official [SageMaker Framework Container for PyTorch 1.3](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html), or the [AWS Deep Learning Container for PyTorch 1.3](https://aws.amazon.com/machine-learning/containers/).

- The library itself supports the following versions when using changes to the training script: PyTorch 1.2, 1.3.

---

## How to Use
### Using Zero Script Change containers
In this case, you don't need to do anything to get the hook running. You are encouraged to configure the hook from the SageMaker python SDK so you can run different jobs with different configurations without having to modify your script. If you want access to the hook to configure certain things which can not be configured through the SageMaker SDK, you can retrieve the hook as follows.
```
import smdebug.pytorch as smd
hook = smd.Hook.create_from_json_file()
```
Note that you can create the hook from smdebug's python API as is being done in the next section even in such containers.

### Bring your own container experience
#### 1. Create a hook
If using SageMaker, you will configure the hook in SageMaker's python SDK using the Estimator class. Instantiate it with
`smd.Hook.create_from_json_file()`. Otherwise, call the hook class constructor, `smd.Hook()`.

#### 2. Register the model to the hook
Call `hook.register_module(net)`.

#### 3. Register your loss function to the hook
If using a loss which is a subclass of `nn.Module`, call `hook.register_loss(loss_criterion)` once before starting training.\
If using a loss which is a subclass of `nn.functional`, call `hook.record_tensor_value(loss)` after each training step.

#### 4. (Optional) Configure Collections, SaveConfig and ReductionConfig
See the [Common API](api.md) page for details on how to do this.

---

## Module Loss Example
```python
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

---

## Functional Loss Example
```python
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

---

## Full API
See the [Common API](api.md) page for details about Collection, SaveConfig, and ReductionConfig.\
See the [Analysis](analysis.md) page for details about analyzing a training job.

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
register_module(
    self,
    module,
)
```
Adds callbacks to the module for recording tensors.

* `module` (torch.nn.Module): The module to use.


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
