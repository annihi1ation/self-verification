import json
import logging
import os
from typing import Callable, List

import yaml
import numpy as np
import sglang as sgl
from sglang.lang.ir import SglFunction
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

from models.verifier import load_verifier
from search.beam_search import beam_search
from search.best_of_n import best_of_n
from search.dvts import dvts
from search.config import SearchConfig
from utils.data import get_dataset, get_questions_from_dataset
from utils.math import prediction_evaluate, majority_voting_evaluate, best_score_evaluate, weighted_sum_evaluate
from utils.parser import H4ArgumentParser
from utils.prompts import build_math_prompt

logging.basicConfig(level=logging.INFO)

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "best_of_n": best_of_n,
    "dvts": dvts,
}


def run_search(args: SearchConfig):
    logger.info("Initializing sglang backend...")
    sgl.set_default_backend(sgl.RuntimeEndpoint(base_url=args.base_url, api_key=args.api_key))

    logger.info(f"Loading approach '{args.approach}' and verifier...")
    approach: SglFunction = APPROACHES[args.approach]
    verifier: Callable[[str, List[str]], List[List[float]]] = load_verifier(args)

    dataset: Dataset = get_dataset(args.dataset_path, args.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.customized_chat_template is not None:
        tokenizer.chat_template = args.customized_chat_template
    args.eos_token = tokenizer.eos_token

    filename_suffix = f"process-{args.process_id}"
    if args.dataset_size is not None:
        dataset = dataset.select(range(args.dataset_size))
        filename_suffix += f"-size-{args.dataset_size}"
    if args.dataset_start is None:
        args.dataset_start = 0
    if args.dataset_end is None:
        args.dataset_end = len(dataset)
    dataset = dataset.select(range(args.dataset_start, args.dataset_end))
    filename_suffix += f"-start-{args.dataset_start}-end-{args.dataset_end}"
    dataset = dataset.select(range(args.process_id, len(dataset), args.num_processes))

    questions = get_questions_from_dataset(dataset)
    prompts = [build_math_prompt(question, tokenizer, system_prompt=args.system_prompt) for question in questions]

    params_list = [dict(question=question, prompt=prompt, verifier=verifier, tokenizer=tokenizer, args=args) for question, prompt in zip(questions, prompts)]
    states = approach.run_batch(params_list, num_threads=args.max_threads, progress_bar=True)

    logger.info("Processing results...")

    dataset = dataset.add_column("prediction", [state["output"] for state in states])
    dataset = dataset.add_column("outputs", [state["outputs"] for state in states])
    dataset = dataset.add_column("scores", [state["scores"] for state in states])
    dataset = dataset.add_column("agg_scores", [state["agg_scores"] for state in states])
    
    dataset = dataset.map(prediction_evaluate, desc="Evaluate prediction")
    dataset = dataset.map(best_score_evaluate, desc="Evaluate best score", fn_kwargs={"n_samples": args.n_samples})

    metrics = [state["metric"] for state in states]
    for metric, data in zip(metrics, dataset):
        for key in data.keys():
            if "label" in key:
                metric[key] = data[key]

    metric = {key: np.mean([m[key] for m in metrics]) for key in metrics[0].keys()}
    metric["num_samples"] = len(dataset)
    dataset.to_json(os.path.join(args.output_dir, f"results-{filename_suffix}.jsonl"), lines=True)
    json.dump(metric, open(os.path.join(args.output_dir, f"metrics-{filename_suffix}.json"), "w"), ensure_ascii=False, indent=4)
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

    logger.info(f"Running search with config: {args}")
    run_search(args)


if __name__ == "__main__":
    main()
