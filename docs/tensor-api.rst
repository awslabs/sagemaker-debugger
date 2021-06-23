Tensor API
----------

An smdebug ``Tensor`` object can be retrieved through the
``trial.tensor(tname)`` API. It is uniquely identified by the string
representing name. It provides the following methods.

.. code:: python

  from smdebug.trials import create_trial
  trial = create_trial(
    path='s3://smdebug-testing-bucket/outputs/resnet',
    name='resnet_training_run'
  )
  output_tensor=trial.tensor(tname)

.. note::
  To use the following methods, you must create a tensor object as shown
  in the code example above. The object name ``output_tensor`` is used as an example
  throughout this page. You can name the object by replacing the ``output_tensor`` as you want.
  For example, if you are logging ``nll_loss_output_0`` outputs from a PyTorch training job,
  you can define the tensor object as following:

  .. code:: python

    output_loss=trial.tensor("nll_loss_output_0")

  You can then replace ``output_tensor`` to ``output_loss`` to use the following methods.

+---------------------------------------------+---------------------------------------+
| Method                                      | Description                           |
+=============================================+=======================================+
| `steps() <#output_tensor.steps>`__          | Query steps for which tensor was      |
|                                             | saved                                 |
+---------------------------------------------+---------------------------------------+
| `value(step) <#output_tensor.value>`__      | Get the value of the tensor at a      |
|                                             | given step as a numpy array           |
+---------------------------------------------+---------------------------------------+
| `reduction_value(step)                      | Get the reduction value of the chosen |
| <#output_tensor.reduction_value>`__         | tensor at a particular step           |
+---------------------------------------------+---------------------------------------+
| `reduction_values                           | Get all reduction values saved for    |
| (step) <#output_tensor.reduction_values>`__ | the chosen tensor at a particular     |
|                                             | step                                  |
+---------------------------------------------+---------------------------------------+
| `values(mode) <#output_tensor.values>`__    | Get the values of the tensor for all  |
|                                             | steps of a given mode                 |
+---------------------------------------------+---------------------------------------+
| `workers(step)                              | Get all the workers for which this    |
| <#output_tensor.workers>`__                 | tensor was saved at a given step      |
+---------------------------------------------+---------------------------------------+
| `prev_steps(step,                           | Get the last n step numbers of a      |
| n) <#output_tensor.prev_steps>`__           | given mode from a given step          |
+---------------------------------------------+---------------------------------------+



.. method:: output_tensor.steps(mode=ModeKeys.GLOBAL, show_incomplete_steps=False)

  Query for the steps at which the given tensor was saved

  **Parameters:**

    - ``mode (smdebug.modes enum value)`` The mode whose steps to return
      for the given tensor. Defaults to ``modes.GLOBAL``
    - ``show_incomplete_steps (bool)`` This parameter is relevant only for
      distributed training. By default this method only returns the steps
      which have been received from all workers. But if this parameter is
      set to True, this method will return steps received from at least one
      worker.

  **Returns:**

    ``list[int]`` A list of steps at which the given tensor was saved

.. method:: output_tensor.value(step_num, mode=ModeKeys.GLOBAL, worker=None)

  Get the value of the tensor at a given step as a numpy array

  **Parameters:**

    - ``step_num (int)`` The step number whose value is to be returned for
      the mode passed through the next parameter.
    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
      number passed above. Defaults to ``modes.GLOBAL``
    - ``worker (str)`` This parameter is only applicable for distributed
      training. You can retrieve the value of the tensor from a specific
      worker by passing the worker name. You can query all the workers seen
      by the trial with the ``trial.workers()`` method. You might also be
      interested in querying the workers which saved a value for the tensor
      at a specific step, this is possible with the method:
      ``trial.tensor(tname).workers(step, mode)``

  **Returns:**

    ``numpy.ndarray`` The value of tensor at the given step and worker (if
    the training job saved data from multiple workers)

.. method:: output_tensor.reduction_value(step_num, reduction_name, mode=modes.GLOBAL, worker=None, abs=False)

  Get the reduction value of the chosen tensor at a particular step. A
  reduction value is a tensor reduced to a single value through reduction
  or aggregation operations. The different reductions you can query for
  are the same as what are allowed in
  `ReductionConfig <api.md#reductionconfig>`__ when saving tensors. This
  API thus allows you to access the reduction you might have saved instead
  of the full tensor. If you had saved the full tensor, it will calculate
  the requested reduction at the time of this call.

  Reduction names allowed are ``min``, ``max``, ``mean``, ``prod``,
  ``std``, ``sum``, ``variance`` and ``l1``, ``l2`` representing the
  norms.

  Each of these can be retrieved for the absolute value of the tensor or
  the original tensor. Above was an example to get the mean of the
  absolute value of the tensor. ``abs`` can be set to ``False`` if you
  want to see the ``mean`` of the actual tensor.

  If you had saved the tensor without any reduction, then you can retrieve
  the actual tensor as a numpy array and compute any reduction you might
  be interested in. In such a case you do not need this method.

  **Parameters:**

    - ``step_num (int)`` The step number whose value is to be returned for
      the mode passed through the next parameter.
    - ``reduction_name (str)`` The name of the reduction to query for. This
      can be one of ``min``, ``max``, ``mean``, ``std``, ``variance``,
      ``sum``, ``prod`` and the norms ``l1``, ``l2``.
    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
      number passed above. Defaults to ``modes.GLOBAL``
    - ``worker (str)`` This parameter is only applicable for distributed
      training. You can retrieve the value of the tensor from a specific
      worker by passing the worker name. You can query all the workers seen
      by the trial with the ``trial.workers()`` method. You might also be
      interested in querying the workers which saved a value for the tensor
      at a specific step, this is possible with the method:
      ``trial.tensor(tname).workers(step, mode)``
    - ``abs (bool)`` If abs is True, this method tries to return the
      reduction passed through ``reduction_name`` after taking the absolute
      value of the tensor. It defaults to ``False``.

  **Returns:**

    ``numpy.ndarray`` The reduction value of tensor at the given step and
    worker (if the training job saved data from multiple workers) as a 1x1
    numpy array. If this reduction was saved for the tensor during training
    as part of specification through reduction config, it will be loaded and
    returned. If the given reduction was not saved then, but the full tensor
    was saved, the reduction will be computed on the fly and returned. If
    both the chosen reduction and full tensor are not available, this method
    raises ``TensorUnavailableForStep`` exception.

.. method:: output_tensor.shape(step_num, mode=modes.GLOBAL, worker=None)

  Get the shape of the chosen tensor at a particular step.

  **Parameters:**

    - ``step_num (int)`` The step number whose value is to be returned for
      the mode passed through the next parameter.
    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
      number passed above. Defaults to ``modes.GLOBAL``
    - ``worker (str)`` This parameter is only applicable for distributed
      training. You can retrieve the value of the tensor from a specific
      worker by passing the worker name. You can query all the workers seen
      by the trial with the ``trial.workers()`` method. You might also be
      interested in querying the workers which saved a value for the tensor
      at a specific step, this is possible with the method:
      ``trial.tensor(tname).workers(step, mode)``

  **Returns:**

    - ``tuple(int)`` If only the shape of this tensor was saved through.
    - ``save_shape`` configuration in ReductionConfig, it will be returned. If
      the full tensor was saved, then shape will be computed and returned
      today. If both the shape and full tensor are not available, this method
      raises ``TensorUnavailableForStep`` exception.

.. method:: output_tensor.values(mode=modes.GLOBAL, worker=None)

  Get the values of the tensor for all steps of a given mode.

  **Parameters:**

    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
      number passed above. Defaults to ``modes.GLOBAL``
    - ``worker (str)`` This parameter is only applicable for distributed
      training. You can retrieve the value of the tensor from a specific
      worker by passing the worker name. You can query all the workers seen
      by the trial with the ``trial.workers()`` method. You might also be
      interested in querying the workers which saved a value for the tensor
      at a specific step, this is possible with the method:
      ``trial.tensor(tname).workers(step, mode)``

  **Returns:**

    ``dict[int -> numpy.ndarray]`` A dictionary with step numbers as keys
    and numpy arrays representing the value of the tensor as values.

.. method:: output_tensor.reduction_values(step_num, mode=modes.GLOBAL, worker=None)

  Get all reduction values saved for the chosen tensor at a particular
  step. A reduction value is a tensor reduced to a single value through
  reduction or aggregation operations. Please go through the description
  of the method ``reduction_value`` for more details.

  **Parameters:**

    - ``step_num (int)`` The step number whose value is to be returned for
      the mode passed through the next parameter.
    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
      number passed above. Defaults to ``modes.GLOBAL``
    - ``worker (str)`` This parameter is only applicable for distributed
      training. You can retrieve the value of the tensor from a specific
      worker by passing the worker name. You can query all the workers seen
      by the trial with the ``trial.workers()`` method. You might also be
      interested in querying the workers which saved a value for the tensor
      at a specific step, this is possible with the method:
      ``trial.tensor(tname).workers(step, mode)``

  **Returns:**

    ``dict[(str, bool) -> numpy.ndarray]`` A dictionary with keys being
    tuples of the form ``(reduction_name, abs)`` to a 1x1 numpy ndarray
    value. ``abs`` here is a boolean that denotes whether the reduction was
    performed on the absolute value of the tensor or not. Note that this
    method only returns the reductions which were saved from the training
    job. It does not compute all known reductions and return them if only
    the raw tensor was saved.

.. method:: output_tensor.shapes(mode=modes.GLOBAL, worker=None)

  Get the shapes of the tensor for all steps of a given mode.

  **Parameters:**

    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
      number passed above. Defaults to ``modes.GLOBAL``
    - ``worker (str)`` This parameter is only applicable for distributed
      training. You can retrieve the value of the tensor from a specific
      worker by passing the worker name. You can query all the workers seen
      by the trial with the ``trial.workers()`` method. You might also be
      interested in querying the workers which saved a value for the tensor
      at a specific step, this is possible with the method:
      ``trial.tensor(tname).workers(step, mode)``

  **Returns:**

    ``dict[int -> tuple(int)]`` A dictionary with step numbers as keys and
    tuples of ints representing the shapes of the tensor as values.

.. method:: output_tensor.workers(step_num, mode=modes.GLOBAL)

  Get all the workers for which this tensor was saved at a given step

  **Parameters:**

    - ``step_num (int)`` The step number whose value is to be returned for
      the mode passed through the next parameter.
    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
      number passed above. Defaults to ``modes.GLOBAL``

  **Returns:**

    ``list[str]`` A list of worker names for which the tensor was saved at
    the given step.

.. method:: output_tensor.prev_steps(step, n, mode=modes.GLOBAL)

  Get the last n step numbers of a given mode from a given step.

  **Parameters:**

    - ``step (int)`` The step number whose value is to be returned for the
      mode passed.
    - ``n (int)`` Number of previous steps to return
    - ``mode (smdebug.modes enum value)`` The mode applicable for the step
    number passed above. Defaults to ``modes.GLOBAL``

  **Returns:**

    ``list[int]`` A list of size at most n representing the previous steps
    for the given step and mode. Note that this list can be of size less
    than n if there were only less than n steps saved before the given step
    in this trial.
