smdebug.analysis
================

smdebug.analysis.utils module
-----------------------------

.. automodule:: smdebug.analysis.utils
   :members:
   :undoc-members:
   :show-inheritance:

Utils
-----

Enable or disable refresh of tensors in a trial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default smdebug refreshes tensors each time you try to query the
tensor. It looks for whether this tensor is saved for new steps and if
so fetches them. If you know the saved data will not change (stopped the
machine learning job), or are not interested in the latest data, you can
stop the refreshing of tensors as follows:

``no_refresh`` takes a trial or a list of trials, which should not be
refreshed. Anything executed inside the with ``no_refresh`` block will
not be refreshed.

.. code:: python

  from smdebug.analysis.utils import no_refresh
  with no_refresh(trials):
      pass

Similarly if you want to refresh tensors only within a block, you can
do:

.. code:: python

  from smdebug.analysis.utils import refresh
  with refresh(trials):
      pass

During rule invocation smdebug waits till the current step is available
and then turns off refresh to ensure that you do not get different
results for methods like ``trial.tensor(name).steps()`` and run into
subtle issues.
