Additional Hook Config APIs
===========================

SaveConfig
----------

The SaveConfig class customizes the frequency of saving tensors. The
hook takes a SaveConfig object which is applied as default to all
tensors included. A collection can also have a SaveConfig object which
is applied to the collection’s tensors. You can also choose to have
different configuration for when to save tensors based on the mode of
the job.

This class is available in the following namespaces ``smdebug`` and
``smdebug.{framework}``.

.. code:: python

   import smdebug as smd
   save_config = smd.SaveConfig(
       mode_save_configs = None,
       save_interval = 100,
       start_step = 0,
       end_step = None,
       save_steps = None,
   )

.. _arguments-1:

Arguments
~~~~~~~~~

-  ``mode_save_configs`` (dict): Used for advanced cases; see details
   below.
-  ``save_interval`` (int): How often, in steps, to save tensors.
   Defaults to 500. A step is saved if ``step % save_interval == 0``
-  ``start_step`` (int): When to start saving tensors.
-  ``end_step`` (int): When to stop saving tensors, exclusive.
-  ``save_steps`` (list[int]): Specific steps to save tensors at. Union
   with save_interval.

Examples
~~~~~~~~

-  ``SaveConfig()`` will save at steps 0, 500, …
-  ``SaveConfig(save_interval=1)`` will save at steps 0, 1, …
-  ``SaveConfig(save_interval=100, end_step=200)`` will save at steps 0,
   100
-  ``SaveConfig(save_interval=100, end_step=201)`` will save at steps 0,
   100, 200
-  ``SaveConfig(save_interval=100, start_step=150)`` will save at steps
   200, 300, …
-  ``SaveConfig(save_steps=[3, 7])`` will save at steps 0, 3, 7, 500, …

Specifying different configuration based on mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is also a more advanced use case, where you specify a different
SaveConfig for each mode. It is best understood through an example:

.. code:: python

   import smdebug as smd
   smd.SaveConfig(mode_save_configs={
       smd.modes.TRAIN: smd.SaveConfigMode(save_interval=1),
       smd.modes.EVAL: smd.SaveConfigMode(save_interval=2),
       smd.modes.PREDICT: smd.SaveConfigMode(save_interval=3),
       smd.modes.GLOBAL: smd.SaveConfigMode(save_interval=4)
   })

Essentially, create a dictionary mapping modes to SaveConfigMode
objects. The SaveConfigMode objects take the same four parameters
(save_interval, start_step, end_step, save_steps) as the main object.
Any mode not specified will default to the default configuration. If a
mode is provided but not all params are specified, we use the default
values for non-specified parameters.

Configuration using SageMaker Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer `Configuring Hook using SageMaker Python
SDK <#configuring-hook-using-sagemaker-python-sdk>`__ and `Configuring
Collection using SageMaker Python
SDK <#configuring-collection-using-sagemaker-python-sdk>`__

--------------

ReductionConfig
---------------

ReductionConfig allows the saving of certain reductions of tensors
instead of saving the full tensor. The motivation here is to reduce the
amount of data saved, and increase the speed in cases where you don’t
need the full tensor. The reduction operations which are computed in the
training process and then saved.

During analysis, these are available as reductions of the original
tensor. Please note that using reduction config means that you will not
have the full tensor available during analysis, so this can restrict
what you can do with the tensor saved. You can choose to also save the
raw tensor along with the reductions if you so desire.

The hook takes a ReductionConfig object which is applied as default to
all tensors included. A collection can also have its own ReductionConfig
object which is applied to the tensors belonging to that collection.

.. code:: python

   import smdebug as smd
   reduction_config = smd.ReductionConfig(
       reductions = None,
       abs_reductions = None,
       norms = None,
       abs_norms = None,
       save_raw_tensor = False,
   )

.. _arguments-2:

Arguments
~~~~~~~~~

-  ``reductions`` (list[str]): Takes names of reductions, choosing from
   “min”, “max”, “median”, “mean”, “std”, “variance”, “sum”, “prod”
-  ``abs_reductions`` (list[str]): Same as reductions, except the
   reduction will be computed on the absolute value of the tensor
-  ``norms`` (list[str]): Takes names of norms to compute, choosing from
   “l1”, “l2”
-  ``abs_norms`` (list[str]): Same as norms, except the norm will be
   computed on the absolute value of the tensor
-  ``save_raw_tensor`` (bool): Saves the tensor directly, in addition to
   other desired reductions

For example,

``ReductionConfig(reductions=['std', 'variance'], abs_reductions=['mean'], norms=['l1'])``

will save the standard deviation and variance, the mean of the absolute
value, and the l1 norm.

.. _configuration-using-sagemaker-python-sdk-1:

Configuration using SageMaker Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reductions are passed as part of the “reductions” parameter to
HookParameters or Collection Parameters. Refer `Configuring Hook using
SageMaker Python SDK <#configuring-hook-using-sagemaker-python-sdk>`__
and `Configuring Collection using SageMaker Python
SDK <#configuring-collection-using-sagemaker-python-sdk>`__ for more on
that.

The parameter “reductions” can take a comma separated string consisting
of the following values:

::

   min
   max
   median
   mean
   std
   variance
   sum
   prod
   l1
   l2
   abs_min
   abs_max
   abs_median
   abs_mean
   abs_std
   abs_variance
   abs_sum
   abs_prod
   abs_l1
   abs_l2
