import json
from datasets import load_dataset


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_dataset(data_path, split="train"):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, split="train")
    elif data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files=data_path, split="train")
    else:
        data = load_dataset(data_path, split=split)
    return data

def get_questions_from_dataset(dataset):
    column_names = dataset.column_names
    for key in ["question", "problem", "prompt"]:
        if key in column_names:
            return dataset[key]
    raise ValueError(f"Cannot parse question keys from dataset {dataset}")
