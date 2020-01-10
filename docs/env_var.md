
## Environment Variables

#### `USE_SMDEBUG`:

When using official [SageMaker Framework Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) and [AWS Deep Learning Containers](https://aws.amazon.com/machine-learning/containers/) which support the [Zero Script Change experience](sagemaker.md#zero-script-change), SageMaker Debugger can be disabled by setting this variable to `0`. In such a case, the hook is disabled regardless of what configuration is given to the job through the SageMaker Python SDK. By default this is set to `1` signifying True.

#### `SMDEBUG_CONFIG_FILE_PATH`:

Contains the path to the JSON file that describes the smdebug hook.

At the minimum, the JSON config should contain the path where smdebug should output tensors.
Example:

`{ "LocalPath": "/my/smdebug_hook/path" }`

In SageMaker environment, this path is set to point to a pre-defined location containing a valid JSON.
In non-SageMaker environment, SageMaker-Debugger is not used if this environment variable is not set and
a hook is not created manually.

Sample JSON from which a hook can be created:
```json
{
  "LocalPath": "/my/smdebug_hook/path",
  "HookParameters": {
    "save_all": false,
    "include_regex": "regex1,regex2",
    "save_interval": "100",
    "save_steps": "1,2,3,4",
    "start_step": "1",
    "end_step": "1000000",
    "reductions": "min,max,mean"
  },
  "CollectionConfigurations": [
    {
      "CollectionName": "collection_obj_name1",
      "CollectionParameters": {
        "include_regex": "regexe5*",
        "save_interval": 100,
        "save_steps": "1,2,3",
        "start_step": 1,
        "reductions": "min"
      }
    },
  ]
}

```

#### `TENSORBOARD_CONFIG_FILE_PATH`:

Contains the path to the JSON file that specifies where TensorBoard artifacts need to
be placed.

Sample JSON file:

`{ "LocalPath": "/my/tensorboard/path" }`

In SageMaker environment, the presence of this JSON is necessary to log any Tensorboard artifact.
By default, this path is set to point to a pre-defined location in SageMaker.

tensorboard_dir can also be passed while creating the hook using the API or
in the JSON specified in SMDEBUG_CONFIG_FILE_PATH. For this, export_tensorboard should be set to True.
This option to set tensorboard_dir is available in both, SageMaker and non-SageMaker environments.


#### `CHECKPOINT_CONFIG_FILE_PATH`:

Contains the path to the JSON file that specifies where training checkpoints need to
be placed. This is used in the context of spot training.

Sample JSON file:

`{ "LocalPath": "/my/checkpoint/path" }`

In SageMaker environment, the presence of this JSON is necessary to save checkpoints.
By default, this path is set to point to a pre-defined location in SageMaker.


#### `SAGEMAKER_METRICS_DIRECTORY`:

Contains the path to the directory where metrics will be recorded for consumption by SageMaker Metrics.
This is relevant only in SageMaker environment, where this variable points to a pre-defined location.


**Note**: The environment variables below are applicable for versions > 0.4.14

#### `SMDEBUG_TRAINING_END_DELAY_REFRESH`:

During analysis, a [trial](analysis.md) is created to query for tensors from a specified directory. This
directory contains collections, events, and index files. This environment variable
specifies how many seconds to wait before refreshing the index files to check if training has ended
and the tensor is available. By default value, this value is set to 1.


#### `SMDEBUG_INCOMPLETE_STEP_WAIT_WINDOW`:

During analysis, a [trial](analysis.md) is created to query for tensors from a specified directory. This
directory contains collections, events, and index files. A trial checks to see if a step
specified in the smdebug hook has been completed. This environment variable
specifies the maximum number of incomplete steps that the trial will wait for before marking
half of them as complete. Default: 1000


#### `SMDEBUG_MISSING_EVENT_FILE_RETRY_LIMIT`:

During analysis, a [trial](analysis.md) is created to query for tensors from a specified directory. This
directory contains collections, events, and index files. All the tensor data is stored in the event files.
When tensor data contained in an event file that is not available has been requested, this variable specifcies
the number of times we retry the request.
