import os
from datasets import load_dataset
from concurrent.futures import TimeoutError
from tqdm import tqdm
from math_verify import parse, verify


def math_verify_evaluate(responses, answers, parse_response=True):
    is_batch = isinstance(responses, list) and isinstance(answers, list)
    if not is_batch:
        responses = [responses]
        answers = [answers]

    scores = []
    for response, answer in zip(responses, answers):
        if parse_response:
            pred = parse(response)
        else:
            pred = response
        gt = parse(f"${answer}$")
        label = verify(gt, pred, strict=False)
        scores.append(label)

    return scores if is_batch else scores[0]
