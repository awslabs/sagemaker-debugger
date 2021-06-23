Release Notes
=============

----

SMDebug Library 1.0.10 Release Notes
-----------------------------------

Date: June 10. 2021


New Features
~~~~~~~~~~~~

- PyTorch

  - Support for PyTorch 1.9.0 (#501)


Bug Fixes
~~~~~~~~~

- None


Improvements
~~~~~~~~~~~~

- TensorFlow

  - Add error handling for TensorFlow v1.x (#498)
  - Add safety checks for TensorFlow v2.x error handling (#497)

- MXNet

  - Add error handling for MXNet (#499)

- XGBoost

  - Add error handling for XGBoost (#496)

- Other

  - Error handling updates for SMDebug to not disrupt training jobs using the default configurations.
    This improvement is to not fail training jobs due to an error in SMDebug or its dependencies.


Known Issues
~~~~~~~~~~~~

- PyTorch

  - The autograd based detailed profiling is not supported for PyTorch 1.9.0.

- SMDebug has a fixed range of framework versions that it supports for TensorFlow and PyTorch.

- Detailed profiling is not supported for training jobs with SageMaker distributed model parallel.


Migration to Deep Learning Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TBD

For previous migrations, see `Release Notes for Deep Learning Containers
<https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html>`__.


----


SMDebug Library Release Notes
-----------------------------

Date: June. 10. 2021

The SMDebug client library started tracking releases.

For previous release notes, see `Releases <https://github.com/awslabs/sagemaker-debugger/releases>`__
in the SMDebug GitHub repository.
