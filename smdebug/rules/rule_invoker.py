# First Party
from smdebug.core.logger import get_logger
from smdebug.exceptions import (
    NoMoreProfilerData,
    RuleEvaluationConditionMet,
    StepUnavailable,
    TensorUnavailable,
    TensorUnavailableForStep,
)

logger = get_logger()


def invoke_rule(rule_obj, start_step=0, end_step=None, raise_eval_cond=False):
    """The rule invoker function against a defined smdebug rule using :class:`~smdebug.rules.Rule`.

    Args:
        rule_obj (Rule): An instance of a subclass of
          :class:`~smdebug.rules.Rule` that you want to invoke.

        start_step (int): Global step number to start invoking the rule
          from. Note that this refers to a global step. The default value is 0.

        end_step (int or  None): Global step number to end the invocation
          of rule before. To clarify, ``end_step`` is an exclusive bound. The
          rule is invoked at ``end_step``. The default value is ``None``, which
          means run till the end of the job.

        raise_eval_cond (bool): This parameter controls whether to raise
          the exception ``RuleEvaluationConditionMet`` when raised by the rule,
          or to catch it and log the message and move to the next step.
          The default value is ``False``, which implies that the it catches the
          exception, logs that the evaluation condition was met for a step and
          moves on to evaluate the next step.

    """
    step = start_step if start_step is not None else 0
    logger.info("Started execution of rule {} at step {}".format(type(rule_obj).__name__, step))
    while (end_step is None) or (step < end_step):
        try:
            rule_obj.invoke(step)
        except (TensorUnavailableForStep, StepUnavailable, TensorUnavailable) as e:
            logger.debug(str(e))
        except RuleEvaluationConditionMet as e:
            # If raise_eval_cond specified, pop up the exception.
            if raise_eval_cond:
                raise e
            else:
                logger.debug(str(e))
                # In case RuleEvaluationConditionMet indicated the end of the rule, break the execution loop.
                if e.end_of_rule:
                    break
        except NoMoreProfilerData as e:
            logger.info(
                "No more profiler data for rule {} at timestamp {}".format(
                    type(rule_obj).__name__, e.timestamp
                )
            )
            break
        step += 1
    # decrementing because we increment step in the above line
    logger.info(
        "Ended execution of rule {} at end_step {}".format(type(rule_obj).__name__, step - 1)
    )
