import numpy as np
import sglang as sgl
from typing import Callable, List
from transformers import PreTrainedTokenizer
from sglang.lang.interpreter import ProgramState

from search.config import SearchConfig
from search.decoding import simple_decoding
from search.utils import add_messages_to_state, aggregate_score


@sgl.function
def best_of_n(s: ProgramState, question, prompt, verifier: Callable[[str, List[str]], List[List[float]]], tokenizer: PreTrainedTokenizer, args: SearchConfig):
    sampling_params = dict(temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = simple_decoding.run_batch([dict(prompt=prompt, **sampling_params) for _ in range(args.n_samples)])
    answers = [output["result"] for output in outputs]
    
    if args.verifier_type == "none":
        scores = [[1.0]] * len(answers)
        agg_scores = aggregate_score(scores, args.agg_strategy)
        best_output = answers[0]
    else:
        scores = verifier(question, answers)
        agg_scores = aggregate_score(scores, args.agg_strategy)
        best_output = answers[np.argmax(agg_scores)]
    
    s["scores"] = scores
    s["agg_scores"] = agg_scores
    s["output"] = best_output
    s["outputs"] = answers
    s["metric"] = {"scores_mean": np.mean([np.mean(sc) for sc in scores]), "agg_scores_mean": np.mean([np.mean(sc) for sc in agg_scores])}
