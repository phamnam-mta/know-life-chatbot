# import logging

# from typing import Any, Text, Dict, List
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.events import SlotSet, Restarted

# from actions.utils.constants import GENERIC_USER, PRONOUNS, PRONOUNS_DEFAULT
# from actions.utils.smartcare_util import get_user_info

# logger = logging.getLogger(__name__)


# class ActionGreet(Action):

#     def name(self) -> Text:
#         return "action_greet"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         try:
#             user_info = get_user_info(tracker)
#             if user_info:
#                 username = user_info.name.title()
#                 gender = user_info.gender
#                 pronouns = PRONOUNS.get(gender, PRONOUNS_DEFAULT)
#                 dispatcher.utter_message(response="utter_greet", pronouns=pronouns, username=username)
#                 return [Restarted(), SlotSet("username", username), SlotSet("pronouns", pronouns)]
#             else:
#                 dispatcher.utter_message(response="utter_something_wrong")
#                 return []
#         except Exception as ex:
#             logger.exception(ex)
#             return []