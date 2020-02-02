import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge


def bleu_compute(ground_truth_utter, answer_sample):
    ground_truth_utter_list = ground_truth_utter.split()
    answer_sample_list = answer_sample.split()
    return sentence_bleu([ground_truth_utter_list], answer_sample_list, smoothing_function=SmoothingFunction().method7,
                         weights=[1./3, 1./3, 1./3])


def rouge_compute(ground_truth_utter, answer_sample):
    rouge = Rouge()
    scores = rouge.get_scores(ground_truth_utter, answer_sample)
    return np.array([scores[0]["rouge-l"]["p"], scores[0]["rouge-l"]["r"], scores[0]["rouge-l"]["f"]])


def rouge_names():
    return ["ROUGE-L Precision", "ROUGE-L Recall", "ROUGE-L F1"]
