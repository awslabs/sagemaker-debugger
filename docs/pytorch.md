# PyTorch

## Contents
- [Support](#support)
- [Module Loss Example](#module-loss-example)
- [Functional Loss Example](#functional-loss-example)
- [Full API](#full-api)

## Support
- SageMaker Zero-Code-Change supported containers: PyTorch 1.3. See [AWS Docs](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html) for more information.
- Python API supported versions: 1.2, 1.3.

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
