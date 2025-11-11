import numpy as np

from typing import List, Dict
from utils.evaluator import math_verify_evaluate
from math_verify import parse, verify
from utils.dapo_extractor import last_boxed_only_string, remove_boxed
from collections import Counter

def prediction_evaluate(example: Dict):
    solution = example["prediction"]
    answer = example["answer"]
    example["label"] = math_verify_evaluate(solution, answer)
    return example


def majority_voting_evaluate(example: Dict, n_samples: int):
    n_subsets = [2**i for i in range(n_samples) if 2**i <= n_samples]
    outputs = example["outputs"]

    for n_subset in n_subsets:
        outputs_subset = outputs[:n_subset]
        preds = [last_boxed_only_string(output) for output in outputs_subset]
        preds = [f"${pred}$" for pred in preds if pred is not None]
        pred_counter = Counter(preds)
        
        # Find the most frequent element in the counter
        if pred_counter:
            pred = max(pred_counter, key=pred_counter.get)
        else:
            pred = ""
        example[f"majority_vote_output@{n_subset}"] = [str(ele) for ele in pred]
        example[f"majority_vote_label@{n_subset}"] = math_verify_evaluate(pred, example["answer"])
    return example


def best_score_evaluate(example: Dict, n_samples: int):
    n_subsets = [2**i for i in range(n_samples) if 2**i <= n_samples]
    outputs = example["outputs"]
    agg_scores = example["agg_scores"]
    
    for n_subset in n_subsets:
        outputs_subset = outputs[:n_subset]
        agg_scores_subset = agg_scores[:n_subset]
        
        pred = outputs_subset[np.argmax(agg_scores_subset)]
        example[f"best_score_output@{n_subset}"] = [str(ele) for ele in parse(pred)]
        example[f"best_score_label@{n_subset}"] = math_verify_evaluate(pred, example["answer"])
    return example


def weighted_sum_evaluate(example: Dict, n_samples: int, score_weight: float = 1.0):
    n_subsets = [2**i for i in range(n_samples) if 2**i <= n_samples]
    outputs = example["outputs"]
    agg_scores = example["agg_scores"]

    for n_subset in n_subsets:
        outputs_subset = outputs[:n_subset]
        agg_scores_subset = agg_scores[:n_subset]
        
        scores_dict = {}
        for output, agg_score in zip(outputs_subset, agg_scores_subset):
            pred = last_boxed_only_string(output)
            if pred is not None:
                pred = f"${pred}$"
                scores_dict[pred] = scores_dict.get(pred, 0) + (1 + agg_score * score_weight)
        
        # Find all elements with the maximum score
        if scores_dict:
            pred = max(scores_dict, key=scores_dict.get)
        else:
            pred = ""
        example[f"weighted_sum_output@{n_subset}"] = pred
        example[f"weighted_sum_label@{n_subset}"] = math_verify_evaluate(pred, example["answer"])
    return example
