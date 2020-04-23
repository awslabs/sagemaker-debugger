# Distributed Training

Sagemaker debugger can be used to debug and save tensors with distributed training.

Here is a list of distributed training strategies that are supported by Sagemaker Debugger.

- [Tensorflow 1.x](#tensorflow-1)
  * [Horovod](#horovod)
  * [MirroredStrategy](#mirrored-strategy)
- [Tensorflow 2.x](#tensorflow-2)
  * [Horovod](#horovod)
  * [MirroredStrategy](#mirrored-strategy)
- [Pytorch](#pytorch)
  * [Horovod](#horovod)
  * [torch.distributed](#torch-distributed)
- [MXNet](#mxnet)
  * [Horovod](#horovod)
- [XGBoost](#xgboost)
  * [Rabit](#rabit)

  ## API

  See [Full API Docs](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md) for the detailed documentation of Sagemaker Debugger API.

Here we highlight distributed training specific API.

- [Creating a hook](#creating-a-hook)
* [Hook from Python constructor](#hook-from-python-constructor)
* [Zero Code Change Hook Initialization](#zero-code-change-hook-initialization)
- [Analysis of saved data](#analysis-of-saved-data)

### Creating A Hook

#### Hook from Python constructor

See [link](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#hook-from-python-constructor) for detailed description.

HookClass below can be one of KerasHook, SessionHook, EstimatorHook for TensorFlow, or is just Hook for MXNet, Pytorch and XGBoost.

```
 hook = HookClass(
    out_dir,
    export_tensorboard = False,
    tensorboard_dir = None,
    dry_run = False,
    reduction_config = None,
    save_config = None,
    include_regex = None,
    include_collections = None,
    save_all = False,
    include_workers="one"
)
```

Arguments:

Full list of arguments are available [here](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#arguments)


...
- `include_workers` (str): Used for distributed training. It can take the values one or all. one means only the tensors from one chosen worker will be saved. This is the default behavior. all means tensors from all workers will be saved.

#### Zero Code Change Hook Initialization

If you are running your training script on a [DLC container](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html).

To do this, you will need to:

1. Create a file with a hook specification.

Here is a minimal hook specification.

```
vim smdebug_hook_config.json

----
{
  "LocalPath": "/tmp/hvd_trial_data",
  "HookParameters":{
    "include_workers": "all",
    "save_interval": 500,
  }
}
```

2. Set the following environment variable

```
export SMDEBUG_CONFIG_FILE_PATH=./smdebug_hook_config.json
```


### Analysis of saved data

See [link](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md) for detailed description.

- [Trial API](#trial-api)
- [Tensor API](#tensor-api)


#### Trial API

For a full list of the Trial API, see [link](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#trial-api)

...
##### workers

Query for all the worker processes from which data was saved by smdebug during multi worker training.

```
trial.workers()
```

###### Returns
`list[str]` A sorted list of names of worker processes from which data was saved. If using TensorFlow Mirrored Strategy for multi worker training, these represent names of different devices in the process. For Horovod, torch.distributed and similar distributed training approaches, these represent names of the form worker_0 where 0 is the rank of the process.


#### Tensor API

For a full list of the Trial API, see [link](https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#tensor-1)

...
##### workers

Get all the workers for which this tensor was saved at a given step

```python
trial.tensor(name).workers(step_num, mode=modes.GLOBAL)
```

###### Arguments
- `name (str)` The name of the tensor to be fetched
- `step_num (int)` The step number whose value is to be returned for the mode passed through the next parameter.
- `mode (smdebug.modes enum value)` The mode applicable for the step number passed above. Defaults to `modes.GLOBAL`

###### Returns
`list[str]` A list of worker names for which the tensor was saved at the given step.


## Tensorflow 1

### Horovod

- [Custom Hook Initializaton](#custom-hook-initializaton)
- [Zero Code Change Hook](#zero-code-change-hook)

#### Custom Hook Initializaton

See full [source-code](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/local/horovod_mnist.py)

Example of minimal hook configuration requried:

```
...
    smd_hook = smd.SessionHook(
        out_dir=args.out_dir,
        save_config=smd.SaveConfig(save_interval=args.save_interval),
        include_collections=["weights", "gradients"],
        include_workers="one"
    )

    ##### Enabling SageMaker Debugger ###########
    # wrapping optimizer so hook can identify gradients
    opt = smd_hook.wrap_optimizer(opt)
...
```

The [example](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/local/horovod_mnist.py) program provided can be executed with the command:

```
mpirun -np 4 -H localhost:4 python tensorflow_mnist.py --out_dir ./hvd_trial_data --save_interval 500 --include_workers all
```
#### Zero Code Change Hook
For details on hook initialization, see [here](#zero-code-change-hook-initialization).

You can run any of the examples provided [here](https://github.com/awslabs/sagemaker-debugger/tree/master/examples/tensorflow/sagemaker_official_container) with the above method.

### Mirrored Strategy

#### Custom Hook Initializaton

See full [source-code](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow/local/horovod_mnist.py)

Example of minimal hook configuration requried:


```
...
    smd_hook = smd.SessionHook(
        out_dir=args.out_dir,
        save_config=smd.SaveConfig(save_interval=args.save_interval),
        include_collections=["weights", "gradients"],
        include_workers="one"
    )

    ##### Enabling SageMaker Debugger ###########
    # wrapping optimizer so hook can identify gradients
    opt = smd_hook.wrap_optimizer(opt)
...
```

The example script can be run with the following command:

```
python tf_mirrored_strategy.py --out_dir ./mirrored_strategy_trial_data --include_workers all
```

To run the same script in Zero Code Change mode


Setup the hook configuration json by following [these steps](#zero-code-change-hook-initialization)

and then run

```
python tf_mirrored_strategy.py --zcc True
```

## Tensorflow 2

### Horovod

- [Custom Hook Initializaton](#custom-hook-initializaton)
  - [Gradient Tape](#gradient-tape)
  - [Fit API Hook](#fit-api)
- [Zero Code Change Hook](#zero-code-change-hook)

#### Custom Hook Initializaton

##### Gradient Tape

See full [source-code](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf2_hvd_gradienttape_mnist.py)


Example of minimal hook configuration requried:

Hook initialization:

```
    smd_hook = smd.KerasHook(
        out_dir=args.out_dir,
        save_config=smd.SaveConfig(save_interval=args.save_interval),
        include_collections=["weights", "gradients"],
        include_workers=args.include_workers
    )
```

Wrap gradient tape with smdebug hook

```
...
    with smd_hook.wrap_tape(tf.GradientTape()) as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)
...
```

Wrap the optimizer so hook can identify gradients
```
    opt = smd_hook.wrap_optimizer(opt)
```

The [example](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf2_hvd_gradienttape_mnist.py) program provided can be executed with the command:

```
mpirun -np 4 -H localhost:4 python tf2_hvd_mnist.py --out_dir ./hvd_trial_data --include_workers all
```

##### Fit API

Example of minimal hook configuration requried:

Initialize the hook object

```
  smd_hook = smd.KerasHook(
      out_dir=args.out_dir,
      save_config=smd.SaveConfig(save_interval=args.save_interval),
      include_collections=["weights", "gradients"],
      include_workers=args.include_workers
  )
```

Wrap the optimizer so hook can identify gradients:

```
opt = smd_hook.wrap_optimizer(opt)
```

The [example](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf2_hvd_keras_fit.py) program provided can be executed with the command:

```
mpirun -np 4 -H localhost:4 python tf2_hvd_keras_fit.py --out_dir ./hvd_trial_data --include_workers all
```

#### Zero Code Change Hook
To run the same script in Zero Code Change mode

Setup the hook configuration json by following [these steps](#zero-code-change-hook-initialization)


### Mirrored Strategy

See full [source-code](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/tensorflow2/scripts/tf_mirrored_strategy.py)

Example of minimal hook configuration requried:


```
...
   hook = smd.KerasHook(
       out_dir=trial_dir,
       save_config=save_config,
       reduction_config=reduction_config,
       include_collections=include_collections,
       save_all=save_all,
       include_workers=include_workers,
   )

   ##### Enabling SageMaker Debugger ###########
   # wrapping optimizer so hook can identify gradients
   opt = smd_hook.wrap_optimizer(opt)
...
```

The example script can be run with the following command:

```
python tf_mirrored_strategy.py --out_dir ./trial_mirrored --include_workers all
```

#### Zero Code Change Hook
To run the same script in Zero Code Change mode

Setup the hook configuration json by following [these steps](#zero-code-change-hook-initialization)

and then run

```
python <example_script>.py --zcc True
```

## Pytorch

### Horovod

- [Custom Hook Initializaton](#custom-hook-initializaton)
- [Zero Code Change Hook](#zero-code-change-hook)

#### Custom Hook Initializaton

See full [source-code](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/pytorch/scripts/horovod_mnist.py).

Minimal custom hook configuration

```
    smd_hook = smd.Hook(
        out_dir=args.out_dir,
        save_config=smd.SaveConfig(save_interval=1),
        include_collections=["weights", "gradients"],
        include_workers=args.include_workers,
    )
```

Register the model with the smdebug hook

```
smd_hook.register_module(model)
```

The example script can be run with the following command:

```
mpirun -np 4 -H localhost:4 python horovod_mnist.py --out_dir ./hvd_mnist_trial --include_workers all
```

#### Zero Code Change Hook

For details on hook initialization, see [here](#zero-code-change-hook-initialization).

You can run this [example](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/pytorch/zero_code_change_examples/horovod_mnist.py) after setting up the hook.

### torch distributed

- [Custom Hook Initializaton](#custom-hook-initializaton)
- [Zero Code Change Hook](#zero-code-change-hook)

#### Custom Hook Initializaton

See full [source-code](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/pytorch/scripts/pytorch_distributed_training.py).


Minimal custom hook configuration

```
    hook = smd.Hook(
        out_dir=out_dir,
        save_config=smd.SaveConfig(save_steps=[0, 1, 5]),
        save_all=True,
        include_workers=include_workers,
    )
```

Register the model with the smdebug hook

```
    hook.register_module(model)
```

The example script can be run with the following command:

```
python pytorch_distributed_training.py --out_dir ./dist_training_trial --include_workers all
```

#### Zero Code Change Hook

For details on hook initialization, see [here](#zero-code-change-hook-initialization).

The example script can be run with the following command after following the above steps

```
python pytorch_distributed_training.py --out_dir ./dist_training_trial --include_workers all --zcc True
```

## MXNet

### Horovod

- [Custom Hook Initializaton](#custom-hook-initializaton)
- [Zero Code Change Hook](#zero-code-change-hook)

#### Custom Hook Initializaton

See full [source-code](https://github.com/awslabs/sagemaker-debugger/blob/master/examples/mxnet/scripts/horovod_mnist.py).

Minimal custom hook configuration

```
    smd_hook = smd.Hook(
        out_dir=args.out_dir,
        save_config=smd.SaveConfig(save_interval=1),
        include_collections=["weights", "gradients"],
        include_workers=args.include_workers,
    )
```

Register the hook with the model

```
hook.register_hook(model)
```

The example script can be run with the following command:

```
mpirun -np 4 -H localhost:4 python horovod_mnist.py --out_dir ./hvd_mnist_trial --include_workers all
```


#### Zero Code Change Hook

For details on hook initialization, see [here](#zero-code-change-hook-initialization).

The example script can be run with the following command after following the above steps

```
mpirun -np 4 -H localhost:4 python horovod_mnist.py --zcc
```

## XGBoost

### Rabit

[TODO]
