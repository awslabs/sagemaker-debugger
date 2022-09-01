Exceptions
----------

SMDebug is designed to be aware of that tensors required to evaluate a rule
may not be available at every step. Hence, it raises a few exceptions
which allow us to control what happens when a tensor is missing. These
are available in the ``smdebug.exceptions`` module. You can import them
as follows:

.. code:: python

  from smdebug.exceptions import *

The following functions are the exceptions (along with others) and their meanings.

.. automodule:: smdebug.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
