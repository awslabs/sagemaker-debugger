# Standard Library
import json

# First Party
from smdebug.core.logger import get_logger

# Local
from .message_action import MessageAction
from .stop_training_action import StopTrainingAction

ALLOWED_ACTIONS = ["stoptraining", "sms", "email"]


class Actions:
    def __init__(self, actions_str, rule_name):
        self._actions = []
        self._logger = get_logger()
        actions_str = actions_str.strip() if actions_str is not None else ""
        if actions_str == "":
            self._logger.info(f"No action specified for rule {rule_name}.")
            return
        self._register_actions(actions_str, rule_name)

    def _register_actions(self, actions_str="", rule_name=""):

        actions_str = actions_str.lower()
        self._logger.info(f"Action string: {actions_str} and rule_name:{rule_name}")
        action_json = json.loads(actions_str)
        actions_list = []
        if isinstance(action_json, dict):
            actions_list.append(action_json)
        elif isinstance(action_json, list):
            actions_list = action_json
        else:
            self._logger.info(
                f"Action string: {actions_str}, expected either a list of dict or dict. Skipping action registering"
            )
            return

        # action : {name:'StopTraining', 'training_job_prefix':''}
        # {name:'sms or email', 'endpoint':''}
        for action_dict in actions_list:
            if not isinstance(action_dict, dict):
                self._logger.info(
                    f"expected dictionary for action, got {action_dict} . Skipping this action."
                )
                continue
            if "name" in action_dict:
                if action_dict["name"] == "stoptraining":
                    training_job_prefix = (
                        action_dict["training_job_prefix"]
                        if "training_job_prefix" in action_dict
                        else None
                    )
                    if training_job_prefix is None:
                        self._logger.info(
                            f"Action :{action_dict['name']} requires 'training_job_prefix'  key to be specified. "
                            f"Action_dict is: {action_dict}"
                        )
                        continue

                    action = StopTrainingAction(rule_name, training_job_prefix)
                    self._actions.append(action)
                elif action_dict["name"] == "sms" or action_dict["name"] == "email":
                    endpoint = action_dict["endpoint"] if "endpoint" in action_dict else None
                    if endpoint is None:
                        self._logger.info(
                            f"Action :{action_dict['name']} requires endpoint key parameter. "
                        )
                        continue

                    action = MessageAction(rule_name, action_dict["name"], endpoint)
                    self._actions.append(action)

                else:
                    self._logger.info(
                        f"Action :{action_dict['name']} not supported. Allowed action names are: {ALLOWED_ACTIONS}"
                    )

    def invoke(self, message=""):
        self._logger.info("Invoking actions")
        for action in self._actions:
            action.invoke(message)
