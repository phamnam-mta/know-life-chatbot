import os
from vncorenlp import VnCoreNLP

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VN_CORE_NLP_PATH = os.path.join(ROOT_DIR, 'utils/vncorenlp/VnCoreNLP-1.1.1.jar')

rdrsegmenter = VnCoreNLP(VN_CORE_NLP_PATH, annotators="wseg", max_heap_size='-Xmx500m') 


def sentences_seg(text, w_seg=True):
	results = []
	sentences = rdrsegmenter.tokenize(text)
	for sent in sentences:
		x = " ".join(sent)
		if not w_seg:
			x = x.replace("_", " ")
		results.append(x)
	return results

def words_seg(text):
	results = []
	sentences = rdrsegmenter.tokenize(text)
	for sent in sentences:
		x = " ".join(sent)
		results.append(x)
	return results
    