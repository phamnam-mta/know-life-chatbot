from typing import Any, Dict, List, Text

import regex
import re
from unicodedata import normalize as nl

import rasa.shared.utils.io
import rasa.utils.io
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES, MESSAGE_ATTRIBUTES
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    RESPONSE_IDENTIFIER_DELIMITER,
    ACTION_NAME,
)


class VietnameseTokenizer(Tokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    # the following language should not be tokenized using the VietnameseTokenizer
    not_supported_language_list = ["zh", "ja", "th"]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the VietnameseTokenizer framework."""

        super().__init__(component_config)

        self.emoji_pattern = rasa.utils.io.get_emoji_regex()
        self.text_normalized = ""

        if "case_sensitive" in self.component_config:
            rasa.shared.utils.io.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
                docs=DOCS_URL_COMPONENTS,
            )

    def remove_emoji(self, text: Text) -> Text:
        """Remove emoji if the full text, aka token, matches the emoji regex."""
        match = self.emoji_pattern.fullmatch(text)

        if match is not None:
            return ""

        return text

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        text = text.strip()
        text = text.strip(u"\ufeff")
        text = text.strip(u"\u200b\u200b\u200b\u200b\u200b\u200b\u200b")
        text = nl('NFKC', text)
        self.text_normalized = self.text_normalize(text.lower())

        # we need to use regex instead of re, because of
        # https://stackoverflow.com/questions/12746458/python-unicode-regular-expression-matching-failing-with-some-unicode-characters

        # remove 'not a word character' if
        words = regex.sub(
            # there is a space or an end of a string after it
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            " ",
            self.text_normalized,
        ).split()

        words = [self.remove_emoji(w) for w in words]
        words = [w for w in words if w]

        # if we removed everything like smiles `:)`, use the whole text as 1 token
        if not words:
            words = [self.text_normalized]

        tokens = self._convert_words_to_tokens(words, self.text_normalized)

        return self._apply_token_pattern(tokens)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Tokenize the incoming message."""
        for attribute in MESSAGE_ATTRIBUTES:
            if isinstance(message.get(attribute), str):
                if attribute in [INTENT, ACTION_NAME, RESPONSE_IDENTIFIER_DELIMITER]:
                    tokens = self._split_name(message, attribute)
                else:
                    tokens = self.tokenize(message, attribute)

                message.set(TOKENS_NAMES[attribute], tokens)
        message.set(TEXT, self.text_normalized, add_to_output=True)

    def text_normalize(self, text):
        """
        Normalize Vietnamese accents
        """

        text = re.sub(r"òa", "oà", text)
        text = re.sub(r"óa", "oá", text)
        text = re.sub(r"ỏa", "oả", text)
        text = re.sub(r"õa", "oã", text)
        text = re.sub(r"ọa", "oạ", text)
        text = re.sub(r"òe", "oè", text)
        text = re.sub(r"óe", "oé", text)
        text = re.sub(r"ỏe", "oẻ", text)
        text = re.sub(r"õe", "oẽ", text)
        text = re.sub(r"ọe", "oẹ", text)
        text = re.sub(r"ùy", "uỳ", text)
        text = re.sub(r"úy", "uý", text)
        text = re.sub(r"ủy", "uỷ", text)
        text = re.sub(r"ũy", "uỹ", text)
        text = re.sub(r"ụy", "uỵ", text)
        text = re.sub(r"Ủy", "Uỷ", text)

        return text