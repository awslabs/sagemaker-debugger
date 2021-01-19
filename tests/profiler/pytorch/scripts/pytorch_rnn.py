# before running this script
# make sure that smdebug is installed
# set env export SMPROFILER_CONFIG_PATH="/tmp/profilerconfig.json"
# dump this json in file:
#  {
#    "ProfilingParameters": {
#        "ProfilerEnabled": true,
#        "LocalPath": "/tmp/test"
#    }
#  }

# Third Party
import torch
import torch.nn as nn

# First Party
import smdebug.pytorch as smd


class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output


rnn = RNN(50, 20, 10)
save_config = smd.SaveConfig(save_interval=500)
hook = smd.Hook(out_dir="/tmp/smdebug", save_all=True, save_config=save_config)

loss_fn = nn.MSELoss()

hook.register_module(rnn)
# hook.register_module(loss_fn)


batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = torch.randn(batch_size, 50)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 10)

loss = 0
#
for t in range(TIMESTEPS):
    # yes! you can reuse the same network several times,
    # sum up the losses, and call backward!
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()

# F F F F F B B F F B B B FFFFFF F BBB
#
