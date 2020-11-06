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
