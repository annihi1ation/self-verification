import json
import logging
import os
from typing import Callable, List

import yaml
import numpy as np
import sglang as sgl
from sglang.lang.ir import SglFunction
from sglang.lang.interpreter import ProgramState
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from models.verifier import load_verifier
from search.beam_search import beam_search
from search.best_of_n import best_of_n
from search.dvts import dvts
from search.residual_mppi import residual_mppi
from search.config import SearchConfig
from search.utils import aggregate_score
from utils.data import get_dataset, get_questions_from_dataset
from utils.math import prediction_evaluate, majority_voting_evaluate, best_score_evaluate, weighted_sum_evaluate
from utils.parser import H4ArgumentParser
from utils.prompts import build_math_prompt

logging.basicConfig(level=logging.INFO)

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@sgl.function
def relabel_func(s: ProgramState, question, outputs, verifier: Callable[[str, List[str]], List[List[float]]], args: SearchConfig):
    scores = verifier(question, outputs)
    agg_scores = aggregate_score(scores, args.agg_strategy)
    best_output = outputs[np.argmax(agg_scores)]
    s["scores"] = scores
    s["agg_scores"] = agg_scores
    s["output"] = best_output
    s["outputs"] = outputs
    s["metric"] = {"scores_mean": np.mean([np.mean(sc) for sc in scores]), "agg_scores_mean": np.mean([np.mean(sc) for sc in agg_scores])}


def add_or_replace_column(dataset: Dataset, column_name: str, values: List[any]):
    if column_name not in dataset.column_names:
        dataset = dataset.add_column(column_name, values)
    else:
        dataset = dataset.map(lambda example, idx: {column_name: values[idx]}, with_indices=True)
    return dataset


def run_relabel(args: SearchConfig):
    sgl.set_default_backend(sgl.RuntimeEndpoint(base_url=args.base_url, api_key=args.api_key))
    verifier: Callable[[str, List[str]], List[List[float]]] = load_verifier(args)

    dataset: Dataset = get_dataset(args.generation_data_path, args.dataset_split)
    if args.dataset_size is not None:
        dataset = dataset.select(range(args.dataset_size))
    
    run_params = [dict(question=example["problem"], outputs=example["outputs"], verifier=verifier, args=args) for example in dataset]
    states = relabel_func.run_batch(run_params, num_threads=args.max_threads, progress_bar=True)
    # Add columns if they don't exist, otherwise replace them
    dataset = add_or_replace_column(dataset, "prediction", [state["output"] for state in states])
    dataset = add_or_replace_column(dataset, "outputs", [state["outputs"] for state in states])
    dataset = add_or_replace_column(dataset, "scores", [state["scores"] for state in states])
    dataset = add_or_replace_column(dataset, "agg_scores", [state["agg_scores"] for state in states])

    dataset = dataset.map(prediction_evaluate, desc="Evaluate prediction")
    dataset = dataset.map(best_score_evaluate, desc="Evaluate best score", fn_kwargs={"n_samples": args.n_samples})
    dataset = dataset.map(majority_voting_evaluate, desc="Evaluate majority voting", fn_kwargs={"n_samples": args.n_samples})
    dataset = dataset.map(weighted_sum_evaluate, desc="Evaluate weighted sum", fn_kwargs={"n_samples": args.n_samples})

    metrics = [state["metric"] for state in states]
    for metric, data in zip(metrics, dataset):
        for key in data.keys():
            if "label" in key:
                metric[key] = data[key]

    metric = {key: np.mean([m[key] for m in metrics]) for key in metrics[0].keys()}
    metric["num_samples"] = len(dataset)
    dataset.to_json(os.path.join(args.output_dir, f"results-relabeled.jsonl"), lines=True)
    json.dump(metric, open(os.path.join(args.output_dir, f"metrics-relabeled.json"), "w"), ensure_ascii=False, indent=4)
    logger.info(f"Done! Metrics: {metric}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = H4ArgumentParser(SearchConfig)
    args: SearchConfig = parser.parse()
    args.post_process()

    # Save config as yaml
    config_dict = {k: str(v) if isinstance(v, type) else v for k, v in args.__dict__.items()}
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logger.info(f"Running relabeling with config: {args}")
    run_relabel(args)


if __name__ == "__main__":
    main()
