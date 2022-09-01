SMDebug Rules
=============

Rules are the medium by which SageMaker Debugger executes a certain
piece of code regularly on different steps of a training job. A rule is
assigned to a trial and can be invoked at each new step of the trial. It
can also access other trials for its evaluation. You can evaluate a rule
using tensors from the current step or any step before the current step.
Please ensure your logic respects these semantics, else you will get a
``TensorUnavailableForStep`` exception as the data would not yet be
available for future steps.

Use Built-in Rules Officially Provided by SageMaker
---------------------------------------------------

Amazon SageMaker Debugger rules analyze tensors emitted during the training of a model.
Debugger offers the Rule API operation that monitors training job progress and errors
for the success of training your model. For example, the rules can detect whether gradients
are getting too large or too small, whether a model is overfitting or overtraining,
and whether a training job does not decrease loss function and improve.
To see a full list of available built-in rules, see
`List of Debugger Built-in Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`__.
.

Write Custom Rules Within or Outside SageMaker
----------------------------------------------

Writing a rule involves implementing the `Rule
APIs <smdebug.rules.html#smdebug-rules-api>`__. Below, let's start with a
simplified version of a custom VanishingGradient rule.

Step 1: Construct a Rule Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a rule involves first inheriting from the base ``Rule`` class
provided by smdebug. For this example rule here, we do not need to look
at any other trials, so we set ``other_trials`` to None.

.. code:: python

  from smdebug.rules import Rule

  class VanishingGradientRule(Rule):
      def __init__(self, base_trial, threshold=0.0000001):
          super().__init__(base_trial, other_trials=None)
          self.threshold = float(threshold)

Please note that apart from ``base_trial`` and ``other_trials`` (if
required), we require all arguments of the rule constructor to take a
string as value. You can parse them to the type that you want from the
string. This means if you want to pass a list of strings, you might want
to pass them as a comma separated string. This restriction is being
enforced so as to let you create and invoke rules from json using
Sagemaker’s APIs.

Step 2: Create a Function to Invoke at a Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this function you can implement the core logic of what you want to do
with these tensors. It should return a boolean value ``True`` or
``False``, where ``True`` means the rule evaluation condition has been
met. When you invoke these rules through SageMaker, the rule evaluation
ends when the rule evaluation condition is met. SageMaker creates a
Cloudwatch event for every rule evaluation job, which can be used to
define actions that you might want to take based on the state of the
rule.

A simplified version of the actual invoke function for
``VanishingGradientRule`` is below:

.. code:: python

  def invoke_at_step(self, step):
      for tensorname in self.base_trial.tensors(collection='gradients'):
          tensor = self.base_trial.tensor(tensorname)
          abs_mean = tensor.reduction_value(step, 'mean', abs=True)
          if abs_mean < self.threshold:
              return True
          else:
              return False

That’s it, writing a rule is as simple as that.

Step 3: Invoke the Rule
~~~~~~~~~~~~~~~~~~~~~~~

Option 1: Invoking a rule through SageMaker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After you’ve written your rule, you can ask SageMaker to evaluate the
rule against your training job by either using SageMaker Python SDK as

.. code:: python

  estimator = Estimator(
      ...
      rules = Rules.custom(
          name='VGRule',
          image_uri='864354269164.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rule-evaluator:latest',
          instance_type='ml.t3.medium', # instance type to run the rule evaluation on
          source='rules/vanishing_gradient_rule.py', # path to the rule source file
          rule_to_invoke='VanishingGradientRule', # name of the class to invoke in the rule source file
          volume_size_in_gb=30, # EBS volume size required to be attached to the rule evaluation instance
          collections_to_save=[CollectionConfig("gradients")], # collections to be analyzed by the rule
          rule_parameters={
              "threshold": "20.0" # this will be used to initialize 'threshold' param in your rule constructor
          }
  )

If you’re using the SageMaker API directly to evaluate the rule, then
you can specify the rule configuration
`DebugRuleConfigurations <https://docs.aws.amazon.com/sagemaker/latest/dg/API_DebugRuleConfiguration.html>`__
in the CreateTrainingJob API request as:

.. code:: python

  "DebugRuleConfigurations": [
      {
          "RuleConfigurationName": "VGRule",
          "InstanceType": "ml.t3.medium",
          "VolumeSizeInGB": 30,
          "RuleEvaluatorImage": "864354269164.dkr.ecr.us-east-1.amazonaws.com/sagemaker-debugger-rule-evaluator:latest",
          "RuleParameters": {
              "source_s3_uri": "s3://path/to/vanishing_gradient_rule.py",
              "rule_to_invoke": "VanishingGradient",
              "threshold": "20.0"
          }
      }
  ]

Option 2: Invoking a rule outside SageMaker through ``invoke_rule``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might want to invoke the rule locally during development. We provide
a function to invoke rules easily. Refer
`smdebug/rules/rule_invoker.py <../smdebug/rules/rule_invoker.py>`__.
The invoke function has the following syntax. It takes a instance of a
Rule and invokes it for a series of steps one after the other.

.. code:: python

  from smdebug.rules import invoke_rule
  from smdebug.trials import create_trial

  trial = create_trial('s3://smdebug-dev-test/mnist-job/')
  rule_obj = VanishingGradientRule(trial, threshold=0.0001)
  invoke_rule(rule_obj, start_step=0, end_step=None)


Rule API
========

.. currentmodule:: smdebug.rules

.. autoclass:: Rule
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: invoke_rule
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
