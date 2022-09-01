Modes for Tensors
-----------------

Used to enumerate different training phases: ``TRAIN``, ``EVAL``, ``PREDICT``,
and ``GLOBAL``. SMDebug APIs use the ``GLOBAL`` mode by default when mode is explicitly set.
You can use this when you register a SMDebug hook to training scripts
and when you retrieve output tensors from specific training phases.

.. autoclass:: smdebug.modes
  :members:
  :undoc-members:
  :show-inheritance:

There are four mode enums as shown below:

.. code:: python

  import smdebug

  smdebug.modes.TRAIN
  smdebug.modes.EVAL
  smdebug.modes.PREDICT
  smdebug.modes.GLOBAL

The modes enum is also available under the framework hook class,
``smdebug.{framework}.modes``.
