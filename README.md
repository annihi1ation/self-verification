# Self-Verification with LLM Verifier

This is the implementation for the paper "Incentivizing LLMs to Self-Verify Their Answers".

## Installation

Our training code is based on the verl framework. First, please follow the installation guide for verl: https://verl.readthedocs.io/en/latest/start/install.html. We recommend to install `vllm==0.8.2` for reimplementation. 

After installing verl, please also check the installation of math-verify.

```bash
pip install math-verify
```

## GRPO for Self-Verification Models

Our training code is located in the `verl-train` folder. The RL script is located in `scripts/math-rl/grpo-qwen-verifier.sh` and `scripts/math-rl/grpo-r1-verifier.sh` for training the Self-Verification-Qwen-7B model and Self-Verification-R1-1.5B model, respectively. You also need to start the ray server if you want to run with multiple nodes. Please refer to `scripts/ray_start_script.sh` to start the ray server.

Before training, please check the following envs and configs are set correctly in the corresponding training script:

1. `WORKING_DIR` is set to the current dir (same level as the README file).
2. `WANDB_API_KEY` is set to your wandb api key.
3. `MODEL_PATH` is set correctly according to your downloaded model.

Then you can run the training script with the following command:

```bash
bash scripts/math-rl/grpo-qwen-verifier.sh
```

We recommend you to use at least 8 GPUs for good efficiency.


## Test-time Scaling with Self-Verification Models

Our test-time scaling code is located in the `evaluation-kit` folder. 

```bash
cd evaluation-kit
```

The evaluation configs are located in the `evaluation-kit/configs` folder. We use the sglang library for inference during test-time scaling. Please refer to the sglang documentation for installation: https://docs.sglang.ai/start/install.html. A common installation command can be: 

```bash
pip install "sglang[all]>=0.4.6.post4"
```

You also need to convert your trained model into the huggingface format. If you have sucessfully run the training script, it will automatically convert the fsdp checkpoints into the huggingface format. You can also mannually do that by running the following command:

```bash
python scripts/model_utils/convert_final_ckpt.py ${OUTPUT_DIR}
```

where `${OUTPUT_DIR}` is the output directory of the training script (from the `OUTPUT_DIR` env). 

For test-time scaling with self-verification, you can run the following command:

```bash
python run_self_verify.py configs/Qwen-rl-models/qwen-rl-verifier.yaml --model_path=path-to-hf-model --model_name=self-verification --dataset_path=data/math500.jsonl --n_samples=16 
```

We also provide other test-time scaling baselines in the configs folder. You can replace the config file to run other baselines like `configs/Qwen-rl-models/qwen-rl-bon.yaml` for running best-of-N for the following command:

```bash
python run_search.py configs/Qwen-rl-models/qwen-rl-bon.yaml --model_path=path-to-hf-model --model_name=self-verification --dataset_path=data/math500.jsonl --n_samples=16 
```

You can also replace the `dataset_path` to run on other datasets located in the `data` folder.
