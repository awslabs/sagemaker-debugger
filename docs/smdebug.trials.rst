SMDebug Trial
=============

An SMDebug trial is an object which lets you query for tensors for a given training
job, specified by the path where SMDebug's artifacts are saved. Trial is
capable of loading new tensors as soon as they become available from the
given path, allowing you to do both offline as well as real-time
analysis.

Create an SMDebug trial object
------------------------------

Depending on the output path, there are two types of trials you can create: LocalTrial or S3Trial.
The SMDebug library provides the following wrapper method that automatically
creates the right trial.

.. autoclass::  smdebug.trials.create_trial
  :members:
  :undoc-members:
  :show-inheritance:
  :inherited-members:
