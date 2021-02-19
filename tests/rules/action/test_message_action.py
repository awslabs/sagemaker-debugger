# First Party
from smdebug.rules.action.action import Actions
from smdebug.rules.action.message_action import MessageAction
from smdebug.rules.action.stop_training_action import StopTrainingAction


def test_action_stop_training_job():
    action_str = '{"name": "stoptraining" , "training_job_prefix":"training_prefix"}'
    action = Actions(actions_str=action_str, rule_name="test_rule")
    action.invoke()


def test_action_stop_training_job_invalid_params():
    action_str = '{"name": "stoptraining" , "invalid_job_prefix":"training_prefix"}'
    action = Actions(actions_str=action_str, rule_name="test_rule")
    action.invoke()


def test_action_sms():
    action_str = '{"name": "sms" , "endpoint":"+11234567890"}'
    action = Actions(actions_str=action_str, rule_name="test_rule")
    action.invoke()
    sms_action = action._actions[0]
    assert sms_action._last_subscription_response is not None
    assert sms_action._last_send_mesg_response is not None


def test_action_sms_invalid_params():
    action_str = '{"name": "sms" , "invalid":"+11234567890"}'
    action = Actions(actions_str=action_str, rule_name="test_rule")
    action.invoke()


def test_action_email():
    action_str = '{"name": "email" , "endpoint":"abc@abc.com"}'
    action = Actions(actions_str=action_str, rule_name="test_rule")
    action.invoke()
    email_action = action._actions[0]
    assert email_action._last_subscription_response is not None
    assert email_action._last_send_mesg_response is not None


def test_action_email_invalid_params():
    action_str = '{"name": "email" , "invalid":"abc@abc.com"}'
    action = Actions(actions_str=action_str, rule_name="test_rule")
    action.invoke()


def test_invalid_message_action():
    action_str = '{"name": "invalid" , "invalid":"abc@abc.com"}'
    action = Actions(actions_str=action_str, rule_name="test_rule")
    action.invoke()


def test_action_multiple():
    action_str = (
        '[{"name": "stoptraining" , "training_job_prefix":"training_prefix"}, {"name": "email" , '
        '"endpoint":"abc@abc.com"}] '
    )
    action = Actions(actions_str=action_str, rule_name="test_rule")
    actions = action._actions
    assert len(actions) == 2
    stop_action = actions[0]
    email_action = actions[1]
    assert isinstance(stop_action, StopTrainingAction) == True
    assert isinstance(email_action, MessageAction) == True

    assert stop_action._training_job_prefix == "training_prefix"
    assert email_action._protocol == "email"
    assert email_action._topic_name == "SMDebugRules"
    assert email_action._message_endpoint == "abc@abc.com"
    assert email_action._rule_name == "test_rule"

    assert email_action._last_subscription_response is not None
