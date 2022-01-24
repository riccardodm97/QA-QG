"""
This class containts the evaluation scripts for the QA (question answering) and the QG (question generation) tasks
Some of the methods used in the QA evaluation script are directly borrowed from the official evaluation script for SQuAD version 1.1
"""
import torch
import numpy as np
import re
import string

from collections import namedtuple, Counter, defaultdict


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(true, pred):
    return float(normalize_answer(true) == normalize_answer(pred))

def f1_score(precision, recall):
    if precision + recall == 0:
            return 0
    return (2 * precision * recall) / (precision + recall)


def accuracy_precision_recall_text(true, pred):
    
    true_tokens = get_tokens(true)
    pred_tokens = get_tokens(pred)

    common = Counter(true_tokens) & Counter(pred_tokens)
    num_same = float(sum(common.values()))

    len_true, len_pred = len(true_tokens), len(pred_tokens)

    # If either is no-answer, then 1 if they agree, 0 otherwise
    if len_true == 0 or len_pred == 0:
        res = float(true_tokens == pred_tokens)
        return res, res, res
    if num_same == 0:
        return 0.0, 0.0, 0.0

    accuracy = num_same / (len_pred + len_true - num_same)
    precision = num_same / len_pred
    recall = num_same / len_true

    return accuracy, precision, recall


def qa_evaluate(data : dict) -> dict:
    
    metrics = defaultdict(list)

    Record = namedtuple('Record', data.keys())
    data_records = [Record(*t) for t in zip(*(data.values()))]    #each element of the list is an example in 

    for ex in data_records:
        pred_start_char = ex.offsets[ex.pred_start][0]
        pred_end_char = ex.offsets[ex.pred_end][1]

        pred_text : str = ex.context[pred_start_char:pred_end_char] 

        acc, prec, rec = accuracy_precision_recall_text(ex.answer, pred_text)
        f1 = f1_score(prec, rec)
        em = compute_exact(ex.answer, pred_text)
 
        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["em"].append(em)
    
    average_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    start_dist = torch.abs(data['pred_start'].float() - data['true_start'].float()).mean()
    end_dist = torch.abs(data['pred_end'].float() - data['true_end'].float()).mean()

    average_metrics['mean_start_dist'] = start_dist.item()
    average_metrics['mean_end_dist'] = end_dist.item()

    average_metrics['numerical_accuracy_start'] = torch.sum(data['pred_start'] == data['true_start'])/(data['true_start'].size(dim=0))
    average_metrics['numerical_accuracy_end'] = torch.sum(data['pred_end'] == data['true_end'])/(data['true_end'].size(dim=0))

    return average_metrics