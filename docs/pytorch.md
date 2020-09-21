# PyTorch

## Contents
- [Support](#support)
- [How to Use](#how-to-use)
- [Module Loss Example](#module-loss-example)
- [Functional Loss Example](#functional-loss-example)
- [Full API](#full-api)

## Support
### Versions
- Zero Script Change experience where you need no modifications to your training script is supported in the official [AWS Deep Learning Container for PyTorch](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#general-framework-containers).

- The library itself supports the following versions when using changes to the training script: PyTorch 1.2, 1.3, 1.4, 1.5, and 1.6.

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

#### 4. Take actions using the hook APIs

For a full list of actions that the hook APIs offer to construct hooks and save tensors, see [Common hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#common-hook-api) and [PyTorch specific hook API](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#pytorch-specific-hook-api).

---

## Module Loss Example
```python
#######################################
# Creating a hook. Refer `API for Saving Tensors` page for more on this
import smdebug.pytorch as smd
hook = smd.Hook(out_dir=args.out_dir)
#######################################

class Model(nn.Module)
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return F.relu(self.fc(x))

net = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

#######################################
# Register the hook and the loss
hook.register_module(net)
hook.register_loss(criterion)
#######################################

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
#######################################
# Register the hook and the loss
import smdebug.pytorch as smd
hook = smd.Hook(out_dir=args.out_dir)
#######################################

class Model(nn.Module)
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return F.relu(self.fc(x))

net = Model()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

#######################################
# Register the hook
hook.register_module(net)
#######################################

# Training loop, recording the loss at each iteration
for (inputs, labels) in trainloader:
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = F.cross_entropy(outputs, labels)

    #######################################
    # Manually record the loss
    hook.record_tensor_value(tensor_name="loss", tensor_value=loss)
    #######################################

    loss.backward()
    optimizer.step()
```

---

## Full API
See the [API for Saving Tensors](api.md) page for details about Hook, Collection, SaveConfig, and ReductionConfig.
See the [Analysis](analysis.md) page for details about analyzing a training job.
