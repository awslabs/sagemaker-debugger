Hook API
========

Create a Hook
-------------

By using AWS Deep Learning Containers, you can directly run your own
training scripts without any additional effort to make it compatible with
the SageMaker Python SDK. For a detailed developer guide for this, see
`Use Debugger in AWS
Containers <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-container.html>`__.

However, for some advanced use cases where you need access to customized
tensors from targeted parts of a training script, you can manually
construct the hook object. The SMDebug library provides hook classes to
make this process simple and compatible with the SageMaker ecosystem and
Debugger. The high-level workflow is as simple as a 2-step process:

1. Register SMDebug hook to your training script.
2. Run a training job within or outside SageMaker.

   - SageMaker APIs for Debugger are available through the SageMaker Python SDK or API.
   - Run it locally

To capture output tensors from your training model, register SMDebug hooks
to your training scripts.

After choosing a framework and defining the hook object, you need to
embed the hooks into target parts of your training script to retrieve
tensors and to use with the SageMaker Debugger Python SDK.

To learn more about registering the hook to your model on a
framework of your choice, see the following pages.

.. toctree::
   :maxdepth: 1

   tensorflow
   pytorch
   mxnet
   xgboost
   hook-constructor
