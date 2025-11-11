import argparse
import copy
import os
import subprocess
from multiprocessing import Pool

import torch

os.unsetenv("VLLM_USE_MODELSCOPE")
os.unsetenv("LMDEPLOY_USE_MODELSCOPE")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="configs/beam_search_prm.yaml")
    parser.add_argument("--num_processes", type=int, default=torch.cuda.device_count())
    # Capture all remaining arguments
    args, unknown_args = parser.parse_known_args()
    # Convert unknown args back to a list of strings
    args.extra_args = unknown_args
    return args


def run_search(args):
    cmd = [
        "python", 
        "run_search.py",
        args.config_file,
        f"--process_id={args.process_id}",
        f"--num_processes={args.num_processes}"
    ]
    # Append any extra arguments to the command
    cmd.extend(args.extra_args)
    print(f"Run command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.process_id)
    subprocess.run(cmd, env=env)


def main():
    args = parse_args()
    num_procs = torch.cuda.device_count()

    configs = []
    for i in range(num_procs):
        config = copy.deepcopy(args)
        config.process_id = i
        config.num_processes = num_procs
        configs.append(config)

    # Run processes in parallel
    try:
        with Pool(num_procs) as pool:
            pool.map(run_search, configs)
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, terminating workers...")
        pool.terminate()
        pool.join()
        raise

    cmd = ["python", "merge_results.py", args.config_file]
    cmd.extend(args.extra_args)
    print(f"[Merge Results] Run command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
