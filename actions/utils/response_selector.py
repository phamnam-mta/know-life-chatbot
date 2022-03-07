import numpy as np
import json
from typing import List, Text
from thefuzz import process
from sentence_transformers.cross_encoder import CrossEncoder
from rasa_addons.nlu.utils.normalizer import text_normalize
from rasa_addons.nlu.utils.tokenizer import words_seg
from actions.utils.constants import MAX_CANDIDATES

class BERTSelector():
    def __init__(self, model_path="/home/phamnam/Documents/vinbrain/know_life/models/response_selector") -> None:
        with open("/home/phamnam/Documents/vinbrain/know_life/data/raw_format/covid_faq.json") as file:
            self.covid_faq = json.load(file)
        self.questions = [q["question"] for q in self.covid_faq]
        self.answers = [q["answer"] for q in self.covid_faq]
        self.model = CrossEncoder(model_path)

    def _get_candidates(self, question: Text) -> List[Text]:
        candidates = process.extract(question, self.questions, limit=MAX_CANDIDATES)
        candidates = [c[0] for c in candidates]
        return candidates

    def get_answer(self, question: Text) -> List:
        question = " ".join(words_seg(text_normalize(question)))
        candidates = self._get_candidates(question)
        doc_normalized = [" ".join(words_seg(text_normalize(c))) for c in candidates]
        
        model_input = [[question, doc] for doc in doc_normalized]
        pred_scores = self.model.predict(model_input, convert_to_numpy=True, show_progress_bar=True)
        pred_scores_argsort = np.argsort(-pred_scores)
        question_matching = candidates[pred_scores_argsort[0]]
        answer_idx = self.questions.index(question_matching)
        return self.answers[answer_idx]