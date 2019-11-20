# Glossary

The imports assume `import smdebug.{tensorflow,pytorch,mxnet,xgboost} as smd`.

**Hook**: The main interface to use training. This object can be passed as a model hook/callback
in Tensorflow and Keras. It keeps track of collections and writes output files at each step.
- `hook = smd.Hook(out_dir="/tmp/mnist_job")`

**Mode**: One of "train", "eval", "predict", or "global". Helpful for segmenting data based on the phase
you're in. Defaults to "global".
- `train_mode = smd.modes.TRAIN`

**Collection**: A group of tensors. Each collection contains its own save configuration and regexes for
tensors to include/exclude.
- `collection = hook.get_collection("losses")`

**SaveConfig**: A Python dict specifying how often to save losses and tensors.
- `save_config = smd.SaveConfig(save_interval=10)`

**ReductionConfig**: Allows you to save a reduction, such as 'mean' or 'l1 norm', instead of the full tensor.
- `reduction_config = ReductionConfig(reductions=['min', 'max', 'mean'], norms=['l1'])`

**Trial**: The main interface to use when analyzing a completed training job. Access collections and tensors.
- `trial = smd.create_trial(out_dir="/tmp/mnist_job")`

**Rule**: A condition that will trigger an exception and terminate the training job early, for example a vanishing gradient.
