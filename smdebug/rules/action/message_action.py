# Standard Library
import json
import os

# Third Party
import boto3

# First Party
from smdebug.core.logger import get_logger

# action :
# {name:'sms' or 'email', 'endpoint':'phone or emailid'}


class MessageAction:
    def __init__(self, rule_name, message_type, message_endpoint):
        self._topic_name = "SMDebugRules"
        self._logger = get_logger()

        if message_type == "sms" or message_type == "email":
            self._protocol = message_type
        else:
            self._protocol = None
            self._logger.info(
                f"Unsupported message type:{message_type} in MessageAction. Returning"
            )
            return
        self._message_endpoint = message_endpoint

        # Below 2 is to help in tests
        self._last_send_mesg_response = None
        self._last_subscription_response = None

        env_region_name = os.getenv("AWS_REGION", "us-east-1")

        self._sns_client = boto3.client("sns", region_name=env_region_name)

        self._topic_arn = self._create_sns_topic_if_not_exists()

        self._subscribe_mesgtype_endpoint()
        self._logger.info(
            f"Registering messageAction with protocol:{self._protocol} endpoint:{self._message_endpoint} and topic_arn:{self._topic_arn} region:{env_region_name}"
        )
        self._rule_name = rule_name

    def _create_sns_topic_if_not_exists(self):
        try:
            topic = self._sns_client.create_topic(Name=self._topic_name)
            self._logger.info(
                f"topic_name: {self._topic_name} , creating topic returned response:{topic}"
            )
            if topic:
                return topic["TopicArn"]
        except Exception as e:
            self._logger.info(
                f"Caught exception while creating topic:{self._topic_name} exception is: \n {e}"
            )
        return None

    def _check_subscriptions(self, topic_arn, protocol, endpoint):
        try:
            next_token = "random"
            subs = self._sns_client.list_subscriptions()
            sub_array = subs["Subscriptions"]
            while next_token is not None:
                for sub_dict in sub_array:
                    proto = sub_dict["Protocol"]
                    ep = sub_dict["Endpoint"]
                    topic = sub_dict["TopicArn"]
                    if proto == protocol and topic == topic_arn and ep == endpoint:
                        self._logger.info(f"Existing Subscription found: {sub_dict}")
                        return True
                if "NextToken" in subs:
                    next_token = subs["NextToken"]
                    subs = self._sns_client.list_subscriptions(NextToken=next_token)
                    sub_array = subs["Subscriptions"]
                    continue
                else:
                    next_token = None
        except Exception as e:
            self._logger.info(
                f"Caught exception while list subscription topic:{self._topic_name} exception is: \n {e}"
            )
        return False

    def _subscribe_mesgtype_endpoint(self):

        response = None
        try:

            if self._topic_arn and self._protocol and self._message_endpoint:
                filter_policy = {}
                if self._protocol == "sms":
                    filter_policy["phone_num"] = [self._message_endpoint]
                else:
                    filter_policy["email"] = [self._message_endpoint]
                if not self._check_subscriptions(
                    self._topic_arn, self._protocol, self._message_endpoint
                ):

                    response = self._sns_client.subscribe(
                        TopicArn=self._topic_arn,
                        Protocol=self._protocol,  # sms or email
                        Endpoint=self._message_endpoint,  # phone number or email addresss
                        Attributes={"FilterPolicy": json.dumps(filter_policy)},
                        ReturnSubscriptionArn=False,  # True means always return ARN
                    )
                else:
                    response = f"Subscription exists for topic:{self._topic_arn}, protocol:{self._protocol}, endpoint:{self._message_endpoint}"
        except Exception as e:
            self._logger.info(
                f"Caught exception while subscribing endpoint on topic:{self._topic_arn} exception is: \n {e}"
            )
        self._logger.info(f"response for sns subscribe is {response} ")
        self._last_subscription_response = response

    def _send_message(self, message):
        response = None
        message = f"SMDebugRule:{self._rule_name} fired. {message}"
        try:
            if self._protocol == "sms":
                msg_attributes = {
                    "phone_num": {"DataType": "String", "StringValue": self._message_endpoint}
                }
            else:
                msg_attributes = {
                    "email": {"DataType": "String", "StringValue": self._message_endpoint}
                }
            response = self._sns_client.publish(
                TopicArn=self._topic_arn,
                Message=message,
                Subject=f"SMDebugRule:{self._rule_name} fired",
                # MessageStructure='json',
                MessageAttributes=msg_attributes,
            )
        except Exception as e:
            self._logger.info(
                f"Caught exception while publishing message on topic:{self._topic_arn} exception is: \n {e}"
            )
        self._logger.info(f"Response of send message:{response}")
        self._last_send_mesg_response = response
        return response

    def invoke(self, message):
        self._send_message(message)
