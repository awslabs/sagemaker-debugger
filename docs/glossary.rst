Glossary
--------

The following glossary items assume you import the SMDebug framework modules as follows:

- **TensorFlow**

  .. code:: python

    import smdebug.tensorflow as smd

- **PyTorch**

  .. code:: python

    import smdebug.pytorch as smd

- **MXNet**

  .. code:: python

    import smdebug.mxnet as smd

- **XGBoost**

  .. code:: python

    import smdebug.xgboost as smd

Terminologies
~~~~~~~~~~~~~

- **Step**: Step means the work done for one batch by a training job
  (i.e. forward and backward pass). (An exception is with TensorFlow’s
  Session interface, where a step also includes the initialization session
  run calls). SageMaker Debugger is designed in terms of steps. When to
  save data is specified using steps. Also, invocation of Rules is
  on a step-by-step basis.

- **Hook**: The main class to pass as a callback object or to create
  callback functions. It keeps track of collections and writes output
  files at each step. The current hook implementation does not support
  merging tensors from current job with tensors from previous job(s).
  Therefore, ensure that the ``out_dir`` does not exist prior to instantiating
  the ‘Hook’ object. - ``hook = smd.Hook(out_dir="/tmp/mnist_job")``

- **Mode**: One of “train”, “eval”, “predict”, or “global”. Helpful for
  segmenting data based on the phase you’re in. Defaults to “global”. -
  ``train_mode = smd.modes.TRAIN``

- **Collection**: A group of tensors. Each collection contains its
  configuration for what tensors are part of it, and when to save them. -
  ``collection = hook.get_collection("losses")``

- **SaveConfig**: A Python dict specifying how often to save losses and
  tensors. - ``save_config = smd.SaveConfig(save_interval=10)``

- **ReductionConfig**: Allows you to save a reduction, such as ‘mean’ or
  ‘l1 norm’, instead of the full tensor. Reductions are simple floats. -
  ``reduction_config = smd.ReductionConfig(reductions=['min', 'max', 'mean'], norms=['l1'])``

- **Trial**: The main interface to use when analyzing a completed training
  job. Access collections and tensors. See `trials
  documentation <analysis.md>`__. -
  ``trial = smd.create_trial(out_dir="/tmp/mnist_job")``

- **Rule**: A condition to monitor the saved data for. It can trigger an
  exception when the condition is met, for example a vanishing gradient.
  See `rules documentation <analysis.md>`__.
