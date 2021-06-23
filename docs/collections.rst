Tensor Collections
------------------

The construct of a Collection groups tensors together. A Collection is
identified by a string representing the name of the collection. It can
be used to group tensors of a particular kind such as “losses”,
“weights”, “biases”, or “gradients”. A Collection has its own list of
tensors specified by include regex patterns, and other parameters
determining how these tensors should be saved and when. Using
collections enables you to save different types of tensors at different
frequencies and in different forms. These collections are then also
available during analysis so you can query a group of tensors at once.

There are a number of built-in collections that SageMaker Debugger
manages by default. This means that the library takes care of
identifying what tensors should be saved as part of that collection. You
can also define custom collections, to do which there are couple of
different ways.

You can specify which of these collections to save in the hook’s
``include_collections`` parameter, or through the ``collection_configs``
parameter to the ``DebuggerHookConfig`` in the SageMaker Python SDK.

Built in Collections
~~~~~~~~~~~~~~~~~~~~

Below is a comprehensive list of the built-in collections that are
managed by SageMaker Debugger. The Hook identifes the tensors that
should be saved as part of that collection for that framework and saves
them if they were requested.

The names of these collections are all lower case strings.

+------------------------+-----------------------+-----------------------+
| Name                   | Supported by          | Description           |
|                        | frameworks/hooks      |                       |
+========================+=======================+=======================+
| ``all``                | all                   | Matches all tensors   |
+------------------------+-----------------------+-----------------------+
| ``default``            | all                   | It’s a default        |
|                        |                       | collection created,   |
|                        |                       | which matches the     |
|                        |                       | regex patterns passed |
|                        |                       | as ``include_regex``  |
|                        |                       | to the Hook           |
+------------------------+-----------------------+-----------------------+
| ``weights``            | TensorFlow, PyTorch,  | Matches all weights   |
|                        | MXNet                 | of the model          |
+------------------------+-----------------------+-----------------------+
| ``biases``             | TensorFlow, PyTorch,  | Matches all biases of |
|                        | MXNet                 | the model             |
+------------------------+-----------------------+-----------------------+
| ``gradients``          | TensorFlow, PyTorch,  | Matches all gradients |
|                        | MXNet                 | of the model. In      |
|                        |                       | TensorFlow when not   |
|                        |                       | using Zero Script     |
|                        |                       | Change environments,  |
|                        |                       | must use              |
|                        |                       | ``hoo                 |
|                        |                       | k.wrap_optimizer()``. |
+------------------------+-----------------------+-----------------------+
| ``losses``             | TensorFlow, PyTorch,  | Saves the loss for    |
|                        | MXNet                 | the model             |
+------------------------+-----------------------+-----------------------+
| ``metrics``            | TensorFlow’s          | For KerasHook, saves  |
|                        | KerasHook, XGBoost    | the metrics computed  |
|                        |                       | by Keras for the      |
|                        |                       | model. For XGBoost,   |
|                        |                       | the evaluation        |
|                        |                       | metrics computed by   |
|                        |                       | the algorithm.        |
+------------------------+-----------------------+-----------------------+
| ``outputs``            | TensorFlow’s          | Matches the outputs   |
|                        | KerasHook             | of the model          |
+------------------------+-----------------------+-----------------------+
| ``layers``             | TensorFlow’s          | Input and output of   |
|                        | KerasHook             | intermediate          |
|                        |                       | convolutional layers  |
+------------------------+-----------------------+-----------------------+
| ``sm_metrics``         | TensorFlow            | You can add scalars   |
|                        |                       | that you want to show |
|                        |                       | up in SageMaker       |
|                        |                       | Metrics to this       |
|                        |                       | collection. SageMaker |
|                        |                       | Debugger will save    |
|                        |                       | these scalars both to |
|                        |                       | the out_dir of the    |
|                        |                       | hook, as well as to   |
|                        |                       | SageMaker Metric.     |
|                        |                       | Note that the scalars |
|                        |                       | passed here will be   |
|                        |                       | saved on AWS servers  |
|                        |                       | outside of your AWS   |
|                        |                       | account.              |
+------------------------+-----------------------+-----------------------+
| ``optimizer_variables``| TensorFlow’s          | Matches all optimizer |
|                        | KerasHook             | variables, currently  |
|                        |                       | only supported in     |
|                        |                       | Keras.                |
+------------------------+-----------------------+-----------------------+
| ``hyperparameters``    | XGBoost               | `Booster              |
|                        |                       | paramamete            |
|                        |                       | rs <https://docs.aws. |
|                        |                       | amazon.com/sagemaker/ |
|                        |                       | latest/dg/xgboost_hyp |
|                        |                       | erparameters.html>`__ |
+------------------------+-----------------------+-----------------------+
| ``predictions``        | XGBoost               | Predictions on        |
|                        |                       | validation set (if    |
|                        |                       | provided)             |
+------------------------+-----------------------+-----------------------+
| ``labels``             | XGBoost               | Labels on validation  |
|                        |                       | set (if provided)     |
+------------------------+-----------------------+-----------------------+
| ``feature_importance`` | XGBoost               | Feature importance    |
|                        |                       | given by              |
|                        |                       | `g                    |
|                        |                       | et_score() <https://x |
|                        |                       | gboost.readthedocs.io |
|                        |                       | /en/latest/python/pyt |
|                        |                       | hon_api.html#xgboost. |
|                        |                       | Booster.get_score>`__ |
+------------------------+-----------------------+-----------------------+
| ``full_shap``          | XGBoost               | A matrix of (nsmaple, |
|                        |                       | nfeatures + 1) with   |
|                        |                       | each record           |
|                        |                       | indicating the        |
|                        |                       | feature contributions |
|                        |                       | (`SHAP                |
|                        |                       | valu                  |
|                        |                       | es <https://github.co |
|                        |                       | m/slundberg/shap>`__) |
|                        |                       | for that prediction.  |
|                        |                       | Computed on training  |
|                        |                       | data with             |
|                        |                       | `predic               |
|                        |                       | t() <https://github.c |
|                        |                       | om/slundberg/shap>`__ |
+------------------------+-----------------------+-----------------------+
| ``average_shap``       | XGBoost               | The sum of SHAP value |
|                        |                       | magnitudes over all   |
|                        |                       | samples. Represents   |
|                        |                       | the impact each       |
|                        |                       | feature has on the    |
|                        |                       | model output.         |
+------------------------+-----------------------+-----------------------+
| ``trees``              | XGBoost               | Boosted tree model    |
|                        |                       | given by              |
|                        |                       | `trees_to_dataframe(  |
|                        |                       | ) <https://xgboost.re |
|                        |                       | adthedocs.io/en/lates |
|                        |                       | t/python/python_api.h |
|                        |                       | tml#xgboost.Booster.t |
|                        |                       | rees_to_dataframe>`__ |
+------------------------+-----------------------+-----------------------+

Default collections saved
~~~~~~~~~~~~~~~~~~~~~~~~~

The following collections are saved regardless of the hook
configuration.

============== ===========================
Framework      Default collections saved
============== ===========================
``TensorFlow`` METRICS, LOSSES, SM_METRICS
``PyTorch``    LOSSES
``MXNet``      LOSSES
``XGBoost``    METRICS
============== ===========================

If for some reason, you want to disable the saving of these collections,
you can do so by setting end_step to 0 in the collection’s SaveConfig.
When using the SageMaker Python SDK this would look like

.. code:: python

  from sagemaker.debugger import DebuggerHookConfig, CollectionConfig

  hook_config = DebuggerHookConfig(
      s3_output_path='s3://smdebug-dev-demo-pdx/mnist',
      collection_configs=[
          CollectionConfig(name="metrics", parameters={"end_step": 0})
      ]
  )

When configuring the Collection in your Python script, it would be as
follows:

.. code:: python

  hook.get_collection("metrics").save_config.end_step = 0

Creating or retrieving a Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| Function                          | Behavior                          |
+===================================+===================================+
| ``hook.                           | Returns the collection with the   |
| get_collection(collection_name)`` | given name. Creates the           |
|                                   | collection with default           |
|                                   | configuration if it doesn’t       |
|                                   | already exist. A new collection   |
|                                   | created by default does not match |
|                                   | any tensor and is configured to   |
|                                   | save histograms and distributions |
|                                   | along with the tensor if          |
|                                   | tensorboard support is enabled,   |
|                                   | and uses the reduction            |
|                                   | configuration and save            |
|                                   | configuration passed to the hook. |
+-----------------------------------+-----------------------------------+

Properties of a Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| Property                          | Description                       |
+===================================+===================================+
| ``tensor_names``                  | Get or set list of tensor names   |
|                                   | as strings                        |
+-----------------------------------+-----------------------------------+
| ``include_regex``                 | Get or set list of regexes to     |
|                                   | include. Tensors whose names      |
|                                   | match these regex patterns will   |
|                                   | be included in the collection     |
+-----------------------------------+-----------------------------------+
| ``reduction_config``              | Get or set the ReductionConfig    |
|                                   | object to be used for tensors     |
|                                   | part of this collection           |
+-----------------------------------+-----------------------------------+
| ``save_config``                   | Get or set the SaveConfig object  |
|                                   | to be used for tensors part of    |
|                                   | this collection                   |
+-----------------------------------+-----------------------------------+
| ``save_histogram``                | Get or set the boolean flag which |
|                                   | determines whether to write       |
|                                   | histograms to enable histograms   |
|                                   | and distributions in TensorBoard, |
|                                   | for tensors part of this          |
|                                   | collection. Only applicable if    |
|                                   | TensorBoard support is enabled.   |
+-----------------------------------+-----------------------------------+

Methods on a Collection
~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| Method                            | Behavior                          |
+===================================+===================================+
| ``coll.include(regex)``           | Takes a regex string or a list of |
|                                   | regex strings to match tensors to |
|                                   | include in the collection.        |
+-----------------------------------+-----------------------------------+
| ``coll.add(tensor)``              | **(TensorFlow only)** Takes an    |
|                                   | instance or list or set of        |
|                                   | tf.Tensor/tf.Variable             |
|                                   | /tf.MirroredVariable/tf.Operation |
|                                   | to add to the collection.         |
+-----------------------------------+-----------------------------------+
| ``coll.add_keras_layer(lay        | **(tf.keras only)** Takes an      |
| er, inputs=False, outputs=True)`` | instance of a tf.keras layer and  |
|                                   | logs input/output tensors for     |
|                                   | that module. By default, only     |
|                                   | outputs are saved.                |
+-----------------------------------+-----------------------------------+
| ``coll.add_module_tensors(modu    | **(PyTorch only)** Takes an       |
| le, inputs=False, outputs=True)`` | instance of a PyTorch module and  |
|                                   | logs input/output tensors for     |
|                                   | that module. By default, only     |
|                                   | outputs are saved.                |
+-----------------------------------+-----------------------------------+
| ``coll.add_block_tensors(blo      | **(MXNet only)** Takes an         |
| ck, inputs=False, outputs=True)`` | instance of a Gluon block,and     |
|                                   | logs input/output tensors for     |
|                                   | that module. By default, only     |
|                                   | outputs are saved.                |
+-----------------------------------+-----------------------------------+

Configuring Collection using SageMaker Python SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters to configure Collection are passed as below when using the
SageMaker Python SDK.

.. code:: python

   from sagemaker.debugger import CollectionConfig
   coll_config = CollectionConfig(
       name="weights",
       parameters={ "parameter": "value" })

The parameters can be one of the following. The meaning of these
parameters will be clear as you review the sections of documentation
below. Note that all parameters below have to be strings. So any
parameter which accepts a list (such as save_steps, reductions,
include_regex), needs to be given as strings separated by a comma
between them.

::

   include_regex
   save_histogram
   reductions
   save_raw_tensor
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

--------------
