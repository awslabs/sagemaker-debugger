# Standard Library
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
        if message_type == "sms" or message_type == "email":
            self._protocol = message_type
        else:
            self._protocol = None
            # TODO log unsupported message type
            return
        self._message_endpoint = message_endpoint
        self._logger = get_logger()
        self._topic_arn = self._create_sns_topic_if_not_exists()

        self._subscribe_mesgtype_endpoint()
        # TODO log debug topic arn , protocol, mesg endpoint
        self._logger.info(
            f"Registering MessageAction with protocol:{self._protocol} endpoint:{self._message_endpoint} and topic_arn:{self._topic_arn} "
        )

        env_region_name = os.environ["AWS_REGION"]
        self._sns_client = boto3.client("sns", region_name=env_region_name)
        self._rule_name = rule_name

    def _create_sns_topic_if_not_exists(self):
        topic = self._sns_client.create_topic(Name=self.topic_name)
        # TODO log info print topic
        self._logger.info(
            f"topic_name: {self._topic_name} , creating topic returned response:{topic}"
        )
        if topic:
            return topic.arn
        return None

    def _subscribe_mesgtype_endpoint(self):

        response = None
        try:
            if self._topic_arn and self._protocol and self._message_endpoint:
                filter_policy = {}
                if self._protocol == "sms":
                    filter_policy["phone_num"] = [self._message_endpoint]
                else:
                    filter_policy["email"] = [self._message_endpoint]
                response = self._sns_client.subscribe(
                    TopicArn=self._topic_arn,
                    Protocol=self._protocol,  # sms or email
                    Endpoint=self._message_endpoint,  # phone number or email addresss
                    Attributes={
                        "FilterPolicy": str(
                            filter_policy
                        )  # FilterPolicy {"phone_num": [  "+16693008439" ]}
                    },
                    ReturnSubscriptionArn=False,  # True means always return ARN
                )
        except Exception as e:
            self._logger.info(
                f"Caught exception while subscribing endpoint on topic:{self._topic_arn} exception is: \n {e}"
            )
        self._logger.info(f"response for sns subscribe is {response} ")

    def _send_message(self, message):
        response = None
        try:
            if self._protocol == "sms":
                msg_attributes = {
                    "phone_num": {"DataType": "string", "StringValue": self._message_endpoint}
                }
            else:
                msg_attributes = {
                    "email": {"DataType": "string", "StringValue": self._message_endpoint}
                }
            response = self._sns_client.publish(
                TopicArn=self._topic_arn,
                # TargetArn='string',  # this or Topic or Phone
                # PhoneNumber='string',
                Message=message,
                Subject="string",
                # MessageStructure='json',
                MessageAttributes=msg_attributes,
            )
        except Exception as e:
            self._logger.info(
                f"Caught exception while getting publishing message on topic:{self._topic_arn} exception is: \n {e}"
            )
        self._logger.info(f"Response of send message:{response}")

    def invoke(self, message=None):
        self._send_message(message)
