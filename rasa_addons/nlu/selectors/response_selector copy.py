from __future__ import annotations
import copy
import logging
from rasa.nlu.featurizers.featurizer import Featurizer

import numpy as np
import tensorflow as tf

from typing import Any, Dict, Optional, Text, Tuple, Union, List, Type

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data import util
import rasa.shared.utils.io
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.classifiers.diet_classifier import (
    DIET,
    LABEL_KEY,
    LABEL_SUB_KEY,
    SENTENCE,
    SEQUENCE,
    DIETClassifier,
)
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.utils.tensorflow import rasa_layers
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    RANKING_LENGTH,
    RENORMALIZE_CONFIDENCES,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    CONNECTION_DENSITY,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    RETRIEVAL_INTENT,
    USE_TEXT_AS_LABEL,
    CROSS_ENTROPY,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CONCAT_DIMENSION,
    FEATURIZERS,
    CHECKPOINT_MODEL,
    DENSE_DIMENSION,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    SOFTMAX,
)
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,
    RESPONSE_SELECTOR_RESPONSES_KEY,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    RESPONSE_SELECTOR_RANKING_KEY,
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    DEFAULT_TRANSFORMER_SIZE,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    RESPONSE,
    INTENT_RESPONSE_KEY,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
)

from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.models import RasaModel

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class CustomResponseSelector(DIETClassifier):
    """Response selector using supervised embeddings.

    The response selector embeds user inputs
    and candidate response into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the response that did not "win".

    The supervised response selector needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``CountVectorsFeaturizer`` that
    can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout.
    """

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        model: Optional[RasaModel] = None,
        all_retrieval_intents: Optional[List[Text]] = None,
        responses: Optional[Dict[Text, List[Dict[Text, Any]]]] = None,
        sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None,
    ) -> None:
        """Declare instance variables with default values.

        Args:
            config: Configuration for the component.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run.
            index_label_id_mapping: Mapping between label and index used for encoding.
            entity_tag_specs: Format specification all entity tags.
            model: Model architecture.
            all_retrieval_intents: All retrieval intents defined in the data.
            responses: All responses defined in the data.
            finetune_mode: If `True` loads the model with pre-trained weights,
                otherwise initializes it with random weights.
            sparse_feature_sizes: Sizes of the sparse features the model was trained on.
        """
        component_config = config

        # the following properties cannot be adapted for the ResponseSelector
        component_config[INTENT_CLASSIFICATION] = True
        component_config[ENTITY_RECOGNITION] = False
        component_config[BILOU_FLAG] = None

        # Initialize defaults
        self.responses = responses or {}
        self.all_retrieval_intents = all_retrieval_intents or []
        self.retrieval_intent = None
        self.use_text_as_label = False

        super().__init__(
            component_config,
            model_storage,
            resource,
            execution_context,
            index_label_id_mapping,
            entity_tag_specs,
            model,
            sparse_feature_sizes=sparse_feature_sizes,
        )

    @property
    def label_key(self) -> Text:
        """Returns label key."""
        return LABEL_KEY

    @property
    def label_sub_key(self) -> Text:
        """Returns label sub_key."""
        return LABEL_SUB_KEY

    def _load_selector_params(self) -> None:
        self.retrieval_intent = self.component_config[RETRIEVAL_INTENT]
        self.use_text_as_label = self.component_config[USE_TEXT_AS_LABEL]

    def _set_message_property(
        self, message: Message, prediction_dict: Dict[Text, Any], selector_key: Text
    ) -> None:
        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})
        message_selector_properties[
            RESPONSE_SELECTOR_RETRIEVAL_INTENTS
        ] = self.all_retrieval_intents
        message_selector_properties[selector_key] = prediction_dict
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,
            message_selector_properties,
            add_to_output=True,
        )

    def train(self, training_data: TrainingData) -> Resource:
        pass

    def _resolve_intent_response_key(
        self, label: Dict[Text, Optional[Text]]
    ) -> Optional[Text]:
        """Given a label, return the response key based on the label id.

        Args:
            label: predicted label by the selector

        Returns:
            The match for the label that was found in the known responses.
            It is always guaranteed to have a match, otherwise that case should have
            been caught earlier and a warning should have been raised.
        """
        for key, responses in self.responses.items():

            # First check if the predicted label was the key itself
            search_key = util.template_key_to_intent_response_key(key)
            if search_key == label.get("name"):
                return search_key

            # Otherwise loop over the responses to check if the text has a direct match
            for response in responses:
                if response.get(TEXT, "") == label.get("name"):
                    return search_key
        return None

    def process(self, messages: List[Message]) -> List[Message]:
        """Selects most like response for message.

        Args:
            messages: List containing latest user message.

        Returns:
            List containing the message augmented with the most likely response,
            the associated intent_response_key and its similarity to the input.
        """
        for message in messages:
            out = self._predict(message)
            top_label, label_ranking = self._predict_label(out)

            # Get the exact intent_response_key and the associated
            # responses for the top predicted label
            label_intent_response_key = (
                self._resolve_intent_response_key(top_label)
                or top_label[INTENT_NAME_KEY]
            )
            label_responses = self.responses.get(
                util.intent_response_key_to_template_key(label_intent_response_key)
            )

            if label_intent_response_key and not label_responses:
                # responses seem to be unavailable,
                # likely an issue with the training data
                # we'll use a fallback instead
                rasa.shared.utils.io.raise_warning(
                    f"Unable to fetch responses for {label_intent_response_key} "
                    f"This means that there is likely an issue with the training data."
                    f"Please make sure you have added responses for this intent."
                )
                label_responses = [{TEXT: label_intent_response_key}]

            for label in label_ranking:
                label[INTENT_RESPONSE_KEY] = (
                    self._resolve_intent_response_key(label) or label[INTENT_NAME_KEY]
                )
                # Remove the "name" key since it is either the same as
                # "intent_response_key" or it is the response text which
                # is not needed in the ranking.
                label.pop(INTENT_NAME_KEY)

            selector_key = (
                self.retrieval_intent
                if self.retrieval_intent
                else RESPONSE_SELECTOR_DEFAULT_INTENT
            )

            logger.debug(
                f"Adding following selector key to message property: {selector_key}"
            )

            utter_action_key = util.intent_response_key_to_template_key(
                label_intent_response_key
            )
            prediction_dict = {
                RESPONSE_SELECTOR_PREDICTION_KEY: {
                    RESPONSE_SELECTOR_RESPONSES_KEY: label_responses,
                    PREDICTED_CONFIDENCE_KEY: top_label[PREDICTED_CONFIDENCE_KEY],
                    INTENT_RESPONSE_KEY: label_intent_response_key,
                    RESPONSE_SELECTOR_UTTER_ACTION_KEY: utter_action_key,
                },
                RESPONSE_SELECTOR_RANKING_KEY: label_ranking,
            }

            self._set_message_property(message, prediction_dict, selector_key)

            if (
                self._execution_context.should_add_diagnostic_data
                and out
                and DIAGNOSTIC_DATA in out
            ):
                message.add_diagnostic_data(
                    self._execution_context.node_name, out.get(DIAGNOSTIC_DATA)
                )

        return messages

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        if self.model is None:
            return None

        with self._model_storage.write_to(self._resource) as model_path:
            file_name = self.__class__.__name__

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.responses.json", self.responses
            )

            rasa.shared.utils.io.dump_obj_as_json_to_file(
                model_path / f"{file_name}.retrieval_intents.json",
                self.all_retrieval_intents,
            )

        super().persist()


    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> CustomResponseSelector:
        """Loads the trained model from the provided directory."""
        model: CustomResponseSelector = super().load(
            config, model_storage, resource, execution_context, **kwargs
        )

        try:
            with model_storage.read_from(resource) as model_path:
                file_name = cls.__name__
                responses = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.responses.json"
                )
                all_retrieval_intents = rasa.shared.utils.io.read_json_file(
                    model_path / f"{file_name}.retrieval_intents.json"
                )
                model.responses = responses
                model.all_retrieval_intents = all_retrieval_intents
                return model
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource, execution_context)

