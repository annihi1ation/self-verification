import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def build_math_messages(question):
    messages = [
        # {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}, 
        {"role": "user", "content": question}
    ]
    return messages


# def build_verification_messages(question, response):
#     messages = [
#         {
#             "role": "user",
#             "content": f'Please verify the solution step by step. At the end of the solution verification, when you give your final grade, write it in the form "Is the answer correct (Yes/No)? X", where X is either Yes or No.\n\nQuestion: {question}\n\nSolution: {response}',
#         }
#     ]
#     return messages


def build_verification_messages(question, response):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
        {"role": "user", "content": f'Please verify the solution step by step. At the end of the solution verification, when you give your final grade, write it in the form "Is the answer correct (Yes/No)? X", where X is either Yes or No.'},
    ]
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_paths", default=["data/raw/math_level3to5_solver_qwen.json"], nargs="+")
    parser.add_argument("--test_data_paths", default=[], nargs="+")
    parser.add_argument("--local_dir", default="data/rlzero/math-level3to5")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_source", default="math-verify")
    parser.add_argument("--train_data_size", default=None, type=int)

    args = parser.parse_args()

    train_datasets = []
    for train_data_path in args.train_data_paths:
        if os.path.splitext(train_data_path)[1] in [".json", ".jsonl"]:
            train_datasets.append(datasets.load_dataset("json", data_files={"train": train_data_path}, split="train"))
        else:
            train_datasets.append(datasets.load_dataset(train_data_path, split="train"))

    test_datasets = []
    for test_data_path in args.test_data_paths:
        test_datasets.append(datasets.load_dataset("json", data_files={"test": test_data_path}, split="test"))

    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            assert "problem" in example or "question" in example, "problem or question not found in the data keys"
            for key in ["problem", "question"]:
                if key in example:
                    question = example.pop(key)
                    break
            assert "answer" in example or "final_answer" in example, "answer or final_answer not found in the data keys"
            if "answer" in example:
                answer = example.pop("answer")
            else:
                answer = example.pop("final_answer")
            if isinstance(answer, list):
                answer = answer[0]

            if data_source.endswith("verification"):
                if "response" not in example:
                    raise ValueError("response not found in the data")
                response = example.pop("response")
                if "</think>" in response:
                    response = response.split("</think>")[1]
                messages = build_verification_messages(question, response)
                answer = example.pop("score")
            else:
                messages = build_math_messages(question)
                answer = str(answer)

            data = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = datasets.concatenate_datasets(train_datasets)
    if args.train_data_size is not None:
        train_dataset = train_dataset.select(range(args.train_data_size))
    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(function=make_map_fn(data_source=args.data_source, split="train"), with_indices=True, remove_columns=remove_columns)

    for i, test_dataset in enumerate(test_datasets):
        remove_columns = test_dataset.column_names
        if args.data_source.endswith("verification"):
            test_datasets[i] = test_dataset.map(function=make_map_fn(data_source=f"test-{args.data_source}", split="test"), with_indices=True, remove_columns=remove_columns)
        else:
            data_name = os.path.basename(args.test_data_paths[i]).split(".")[0]
            test_datasets[i] = test_dataset.map(function=make_map_fn(data_source=f"math-verify-{data_name}", split="test"), with_indices=True, remove_columns=remove_columns)

    # print examples from train and test data
    print(f"Training data length: {len(train_dataset)}")
    print(f"Training dataset examples: ")
    print(train_dataset[0])
    print(train_dataset[-1])
    print("=" * 50)
    print(f"Test dataset examples: ")
    for i, test_dataset in enumerate(test_datasets):
        print(test_dataset[0])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f'{os.path.basename(args.train_data_paths[0]).split(".")[0]}.parquet'))
    print("Training data saved to {}".format(os.path.join(local_dir, f'{os.path.basename(args.train_data_paths[0]).split(".")[0]}.parquet')))
    for i, (test_data_path, test_dataset) in enumerate(zip(args.test_data_paths, test_datasets)):
        if args.data_source.endswith("verification"):
            save_path = os.path.join(local_dir, f'{os.path.basename(test_data_path).split(".")[0]}.parquet')
            test_dataset.to_parquet(save_path)
            print("Test data saved to {}".format(save_path))
        else:
            save_path = os.path.join(local_dir, f'{os.path.basename(test_data_path).split(".")[0]}.parquet')
            test_dataset.to_parquet(save_path)
            print("Test data saved to {}".format(save_path))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
