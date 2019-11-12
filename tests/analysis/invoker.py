# First Party
from smdebug.core.logger import get_logger
from smdebug.exceptions import *
from smdebug.rules.rule_invoker import create_rule

logger = get_logger()


def invoke_rule(rule_obj, flag, start_step, end_step):
    step = start_step if start_step is not None else 0
    logger.info("Started execution of rule {} at step {}".format(type(rule_obj).__name__, step))
    exception_thrown = "False"
    while (end_step is None) or (
        step < end_step
    ):  # if end_step is not provided, do infinite checking
        try:
            rule_obj.invoke(step)
            step += 1
        except StepUnavailable as e:
            logger.info(e)
            step += 1
        except TensorUnavailableForStep as e:
            logger.info(e)
            step += 1
        except RuleEvaluationConditionMet as e:
            logger.info(e)
            step += 1
            exception_thrown = "True"
            break
        except NoMoreData as e:
            logger.info(e)
            break

    logger.info(
        "Ending execution of rule {} with step={} ".format(rule_obj.__class__.__name__, step - 1)
    )
    msg = "Flag passed :{} , exception_thrown:{}".format(flag, exception_thrown)
    if flag != exception_thrown:
        assert False, msg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rule invoker takes the below arguments and"
        "any argument taken by the rules. The arguments not"
        "mentioned below are automatically passed when"
        "creating the rule objects."
    )
    parser.add_argument("--tornasole_path", type=str, required=True)
    parser.add_argument("--rule_name", type=str, required=True)
    parser.add_argument(
        "--other-trials",
        type=str,
        help="comma separated paths for " "other trials taken by the rule",
    )
    parser.add_argument("--start_step", type=int)
    parser.add_argument("--end_step", type=int)
    parser.add_argument("--flag", type=str, default=None)
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith("--"):
            parser.add_argument(arg, type=str)
    args = parser.parse_args()
    args_dict = vars(args)
    # to standardize args for create_rule function
    args.trial_dir = args.tornasole_path
    r = create_rule(args, args_dict)
    invoke_rule(r, flag=args.flag, start_step=args.start_step, end_step=args.end_step)
