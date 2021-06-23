Configure Hook using SageMaker Python SDK
=========================================

After you make the changes to your training script, you can
configure the hook with parameters to the SageMaker Debugger API
operation, ``DebuggerHookConfig``.

.. code:: python

    from sagemaker.debugger import DebuggerHookConfig

    collection_configs=[
        CollectionConfig(name="tensor_collection_1")
        CollectionConfig(name="tensor_collection_2")
        ...
        CollectionConfig(name="tensor_collection_n")
    ]

    hook_config = DebuggerHookConfig(
        s3_output_path='s3://smdebug-dev-demo-pdx/mnist',
        collection_configs=collection_configs,
        hook_parameters={
           "parameter": "value"
        }
    )

Path to SMDebug artifacts
-------------------------

To create an SMDebug trial object, you need to know where the SMDebug artifacts are saved.

1. For SageMaker training jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running a SageMaker job, SMDebug artifacts are saved to Amazon S3.
SageMaker saves data
from your training job to a local path of the training container and
uploads them to an S3 bucket of your account. When you start a
SageMaker training job with the python SDK, you can set the path
using the parameter ``s3_output_path`` of the ``DebuggerHookConfig``
object. If you don't specify the path, SageMaker automatically sets the
output path to your default S3 bucket.

**Example**

.. code:: python

  from sagemaker.debugger import CollectionConfig, DebuggerHookConfig

  collection_configs=[
      CollectionConfig(name="weights"),
      CollectionConfig(name="gradients")
  ]

  debugger_hook_config=DebuggerHookConfig(
    s3_output_path="specify-your-s3-bucket-uri"  # Optional
    collection_configs=collection_configs
  )

For more information, see `Configure Debugger Hook to Save Tensors
<https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-configure-hook.html>`__
in the *Amazon SageMaker Developer Guide*.

2. For non-SageMaker training jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are running a training job outside SageMaker, this is the path you
pass as ``out_dir`` when you create an SMDebug Hook.
When creating the hook, you can
pass either a local path (for example, ``/home/ubuntu/smdebug_outputs/``)
or an S3 bucket path (for example, ``s3://bucket/prefix``).

Hook Configuration Parameter Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The available ``hook_parameters`` keys are listed in the following. The meaning
of these parameters will be clear as you review the sections of
documentation below. Note that all parameters below have to be strings.
So for any parameter which accepts a list (such as save_steps,
reductions, include_regex), the value needs to be given as strings
separated by a comma between them.

::

   dry_run
   save_all
   include_workers
   include_regex
   reductions
   save_raw_tensor
   save_shape
   save_interval
   save_steps
   start_step
   end_step
   train.save_interval
   train.save_steps
   train.start_step
   train.end_step
   eval.save_interval
   eval.save_steps
   eval.start_step
   eval.end_step
   predict.save_interval
   predict.save_steps
   predict.start_step
   predict.end_step
   global.save_interval
   global.save_steps
   global.start_step
   global.end_step
