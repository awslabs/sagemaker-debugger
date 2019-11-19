# PyTorch

Supported PyTorch versions: 1.2+.

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
