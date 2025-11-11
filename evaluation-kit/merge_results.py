import json
import logging
import os

from search.config import SearchConfig
from utils.data import load_jsonl, save_jsonl
from utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = H4ArgumentParser(SearchConfig)
    args = parser.parse()
    args.post_process()
    logger.info(f"Merging results from {args.output_dir}")

    # Get all results files
    results_files = [f for f in os.listdir(args.output_dir) if f.startswith("results-process") and f.endswith(".jsonl")]
    if len(results_files) == 0:
        logger.info("No results files found")
        return

    # Load and merge all results
    all_results = []
    for file in results_files:
        if file == "results-all.jsonl":
            continue
        results = load_jsonl(os.path.join(args.output_dir, file))
        all_results.extend(results)

    # Save merged results
    output_path = os.path.join(args.output_dir, "results-all.jsonl")
    save_jsonl(all_results, output_path)
    logger.info(f"Saved merged results to {output_path}")

    # Calculate and save overall metrics
    metric_files = [f for f in os.listdir(args.output_dir) if f.startswith("metrics-process") and f.endswith(".json")]
    metrics = {}
    for file in metric_files:
        metric = json.load(open(os.path.join(args.output_dir, file)))
        num_samples = metric.get("num_samples", 1)
        metrics["num_samples"] = metrics.get("num_samples", 0) + num_samples
        for key, value in metric.items():
            if key != "num_samples":
                metrics[key] = metrics.get(key, 0) + value * num_samples
    
    for key in metrics:
        if key != "num_samples":
            metrics[key] /= metrics["num_samples"]

    metrics_path = os.path.join(args.output_dir, "metrics-all.json")
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    logger.info(f"Saved overall metrics to {metrics_path}")
    logger.info(f"Exp metrics: {metrics}")


if __name__ == "__main__":
    main()
