# ParameterServerStrategy MNIST Example
We provide an example script `parameter_server_mnist.py` which is a Tornasole-enabled TensorFlow training script
that uses the ParameterServer to perform distributed training.

It uses the Estimator interface of TensorFlow.

This is an example of how you can log a distributed training job with smdebug.

## Integrating Tornasole
Below we call out the changes for Tornasole in the above script and describe them

**Importing TornasoleTF**
```
import smdebug.tensorflow as ts
```
**Saving gradients**

We need to wrap our optimizer with hook.wrap_optimizer, and use this optimizer to minimize loss.
This will also enable us to access the gradients during analysis without having to identify which tensors out of the saved ones are the gradients.
```
optimizer = hook.wrap_optimizer(optimizer)
```


**Setting save interval**

You can set different save intervals for different modes.
This can be done by passing a dictionary as save_config to the hook.
This dictionary should have the mode as key and a SaveConfig object as value.
```
ts.TornasoleHook(...,
    save_config=ts.SaveConfig(save_interval=FLAGS.tornasole_frequency),
```
**Setting the right mode**

Notice the calls to `hook.set_mode` at various places in the code.
```
hook.set_mode(ts.modes.TRAIN)
```

```
hook.set_mode(ts.modes.EVAL)
```
**Passing the hook**

We need to pass this hook to a monitored session and use this session for running the job.
```
ts_hook = ts.TornasoleHook(...)
mnist_classifier.train(..., hooks=[ts_hook])
```

```
mnist_classifier.evaluate(..., hooks=[ts_hook])
```
## Running the example
### Environment
Ensure you are in a python environment which has tornasole_core installed. If you followed the recommended instructions of using Amazon Deep Learning AMI, then you might want to activate the tensorflow_p36 environment as follows.
```
source activate tensorflow_p36
```
### Tornasole Path
We recommend saving tornasole outputs on S3 by passing the
flag `--tornasole_path` in the format `s3://bucket_name/prefix`.
The commands below will be shown with local path however so you can
run them immediately without having to setup S3 permissions.

### Command

The example script performs distributed training 2 workers and 1 parameter server by default.

The cluster config used by the demo can be overriden by setting the

TF_CONFIG environment variable before running the script. Details for that can be found here. [link](https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config_environment_variable)


The default cluster specification used by the script is given below:

```
        os.environ["TF_CONFIG"] = json.dumps(
            {
                "cluster": {"worker": [nodes[0], nodes[1]], "ps": [nodes[2]]},
                "task": {"type": FLAGS.node_type, "index": FLAGS.task_index},
            }
        )
```

The values of the nodes is populated by this snippet in the script:

```
    try:
        f = open(FLAGS.hostfile)
        for line in f.readlines():
            nodes.append(line.strip())
    except OSError as e:
        print(e.errno)
```

The script uses a hostfile as an input, where each line corresponds to a node.

A sample hostfile can be found here [hostfile.txt](../../../../../examples/tensorflow/scripts/distributed_training/parameter_server_training/hostfile.txt)

To setup the parameter server:

```
python parameter_server_mnist.py \
--hostfile hostfile.txt \
--steps 1000 \
--tornasole_path ~/ts_output/ps_training  \
--tornasole_frequency 100 \
--node_type ps --task_index 0
```

To setup the first worker server:

```
python parameter_server_mnist.py \
--hostfile hostfile.txt \
--steps 1000 \
--tornasole_path ~/ts_output/ps_training  \
--tornasole_frequency 100 \
--node_type worker --task_index 0
```

To setup the second worker server:

```
python parameter_server_mnist.py \
--hostfile hostfile.txt \
--steps 1000 \
--tornasole_path ~/ts_output/ps_training  \
--tornasole_frequency 100 \
--node_type worker --task_index 1
```


Note: You can limit the number of GPUs used by each worker by setting. See [link](https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on)
```
export CUDA_VISIBLE_DEVICES=0,1
```


### Analysis
Refer [this page](../../../../rules/README.md) for more details on analysis.

### More
Please refer to [Tornasole Tensorflow page](../../../README.md).
