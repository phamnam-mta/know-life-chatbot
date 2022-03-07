import json
import logging
from thefuzz import process
from typing import Any, Text, Dict, List
from aio_pika import message

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from actions.utils.response_selector import BERTSelector

logger = logging.getLogger(__name__)

class ActionCovidFAQ(Action):

    def __init__(self) -> None:
        super().__init__()
        self.selector = BERTSelector()

    def name(self) -> Text:
        return "action_covid_faq"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            message = tracker.latest_message.get("text")
            answer = self.selector.get_answer(message)

            dispatcher.utter_message(text=answer)

            return []
        except Exception as ex:
            logger.exception(ex)
            return []
