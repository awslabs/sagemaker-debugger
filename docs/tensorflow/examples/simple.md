# Simple Example
We provide a simple example script `simple.py` which is a Tornasole-enabled TensorFlow training script. It uses the Session interface of TensorFlow. 
Here we show different scenarios of how to use Tornasole to save different tensors during training for analysis. 
Below are listed the changes we made to integrate these different behaviors of Tornasole as well as example commands for you to try.

## Integrating Tornasole
Below we call out the changes for Tornasole in the above script and describe them

**Importing TornasoleTF**
```
import tornasole.tensorflow as ts
```
**Saving all tensors**
```
ts.TornasoleHook(..., save_all=True, ...)
```
**Saving gradients**

We need to wrap our optimizer with TornasoleOptimizer, and use this optimizer to minimize loss. 
This will also enable us to access the gradients during analysis without having to identify which tensors out of the saved ones are the gradients.
```
opt = TornasoleOptimizer(opt)
optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

ts.TornasoleHook(..., include_collections=[..,'gradients'], ...)
```
**Saving losses**

Since we are not using a default loss function from Tensorflow, 
we need to tell Tornasole to add our loss to the losses collection as follows
```
ts.add_to_collection('losses', loss)
``` 
In the code, you will see the following line to do so. 
```
ts.TornasoleHook(..., include_collections=[...,'losses'], ...)
```

**Setting save interval**
```
ts.TornasoleHook(...,save_config=ts.SaveConfig(save_interval=args.tornasole_frequency)...)
```
**Setting the right mode**

Since we are only training here, you will see in the code that the 
appropriate training mode has been set before the session run calls. 
```
hook.set_mode(ts.modes.TRAIN)
```
**Passing the hook**

We need to pass this hook to a monitored session and use this session for running the job.
```
hook = ts.TornasoleHook(...)
sess = tf.train.MonitoredSession(hooks=[hook])
```

## Running the example
### Environment
Ensure you are in a python environment which has TensorFlow, TornasoleTF and TornasoleCore installed. If you followed the recommended instructions of using Amazon Deep Learning AMI, then you might want to activate the tensorflow_p36 environment as follows.
```
source activate tensorflow_p36
```
### Tornasole Path
We recommend saving tornasole outputs on S3 by passing the 
flag `--tornasole_path` in the format `s3://bucket_name/prefix`. 
The commands below will be shown with local path however so you can 
run them immediately without having to setup S3 permissions.
### Example commands

#### Running a well behaved job
```
python simple.py --tornasole_path ~/ts_outputs/ok --lr 0.001 --scale 1 --steps 100 --tornasole_frequency 13
```
This will generate output like:
```
INFO:tornasole:Saving for step 0: 89 objects
INFO:tornasole:Save complete, saved 1462 bytes
Step=0, Loss=83.92036437988281
Step=1, Loss=92.88887786865234
Step=2, Loss=119.52877044677734
Step=3, Loss=63.18230438232422
[...]
INFO:tornasole:Saving for step 91: 89 objects
INFO:tornasole:Save complete, saved 1462 bytes
Step=96, Loss=129.8429412841797
Step=97, Loss=95.37699127197266
Step=98, Loss=89.81304168701172
Step=99, Loss=75.2679214477539

```
Tornasole is saving all tensors every 13 steps (you can customize to save only certain tensors). 
Tensors have been saved in `~/ts_outputs/ok/`.

#### Running a job which produces nan
Now run the same job, but this time injecting errors (large learning rate, incorrect scaling features):
```
python simple.py --tornasole_path ~/ts_outputs/not_good --lr 100 --scale 100000000000 --tornasole_frequency 9 --steps 100
```
This will generate: 
```
INFO:tornasole:Saving for step 0: 89 objects
INFO:tornasole:Save complete, saved 1462 bytes
Step=0, Loss=1.0731928032228293e+24
Step=1, Loss=1.1620568222637874e+24
Step=2, Loss=nan
Step=3, Loss=nan
...
Step=96, Loss=nan
Step=97, Loss=nan
Step=98, Loss=nan
INFO:tornasole:Saving for step 99: 89 objects
INFO:tornasole:Save complete, saved 1462 bytes
Step=99, Loss=nan
```
Tornasole is saving every 9 steps.
Tensors have been saved in `~/ts_outputs/not_good/`. 

### Analysis
We can invoke a rule provided by Tornasole to monitor tensors for nan.
This can be run even while training is going on, it will continuously monitor tensors and
invoke the rule on each new step. Once the training ends you can stop this job.
You can also do the same analysis after the training job has ended. 
```
python -m tornasole.rules.rule_invoker --trial-dir ~/ts_outputs/not_good --rule-name ExplodingTensor
```
Refer [this page](../../rules/README.md) for more details on analysis.

### More
Please refer to [Tornasole Tensorflow page](../README.md) and the various flags in the script to customize the behavior further.
 

