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

### Training Flow

The GRPO training pipeline for Self-Verification Models proceeds through the following stages:

#### Step 1 — Data Preparation (`scripts/math-rl/math_preprocess.py`)

Raw math problem datasets (JSON/JSONL) are converted into Parquet files. Each training sample is formatted as a **multi-turn conversation** where the model sees both the problem and a candidate solution, then is asked to verify it:

```
Turn 1 (User):    <math problem>
Turn 2 (Assistant): <candidate solution>
Turn 3 (User):    "Please verify the solution step by step. At the end of the
                   solution verification, when you give your final grade, write
                   it in the form 'Is the answer correct (Yes/No)? X', where X
                   is either Yes or No."
```

Each sample's metadata includes a `ground_truth` field encoded as `"<label>|<correct_ratio>"`, where `label` is the Python string `'True'`/`'False'` (whether the candidate solution is correct — converted to a bool via `eval()` during reward scoring) and `correct_ratio` is the fraction of correct solutions in that group (used for imbalance-aware reward shaping).

#### Step 2 — Data Loading

`RLHFDataset` tokenizes the multi-turn prompts with the model's chat template, filters out prompts that exceed `max_prompt_length`, and yields batches of `(input_ids, attention_mask, reward_model_metadata)`.

#### Step 3 — Rollout / Generation

At each training step, the current actor policy generates **N = 8 responses** per prompt using vLLM (configurable via `actor_rollout_ref.rollout.n`). Each response is a chain-of-thought verification ending with the verdict `"Is the answer correct (Yes/No)? Yes"` or `"…? No"`.

#### Step 4 — Reward Scoring (`verl/utils/reward_score/math_verification.py`)

The reward function (`compute_score`) checks each generated response against the ground-truth label. Two reward types are supported (selected via the `VERIFICATION_REWARD_TYPE` environment variable):

| Type | Formula |
|---|---|
| `baseline` | `reward = 1.0` if prediction matches label, else `0.0` |
| `fix_imbalance` | When prediction matches label: `reward = 2 × [float(pred) × (1 − correct_ratio) + (1 − float(pred)) × correct_ratio]`; otherwise `0.0`. Here `float(True) = 1.0` and `float(False) = 0.0`, so a correct **Yes** gets weight `2 × (1 − correct_ratio)` and a correct **No** gets weight `2 × correct_ratio`. |

The `fix_imbalance` type up-weights harder-to-predict cases to counteract class imbalance in the verification labels.

Optional **auxiliary penalties** can be added via `AUXILIARY_REWARDS`:
- `non_short_response`: `−0.5` if the response is shorter than 40 characters.
- `no_code`: `−0.5` if the response contains a Python code block.

#### Step 5 — GRPO Advantage Computation (`verl/trainer/ppo/core_algos.py`)

Responses are grouped by their originating prompt. Within each group the scalar rewards are normalized:

```
advantage_i = (score_i − mean(group_scores)) / (std(group_scores) + ε)
```

This is the core Group Relative Policy Optimization (GRPO) mechanism: no separate critic network is needed, and the advantage is derived purely from the relative quality of responses within a group.

#### Step 6 — Policy Update

The actor is updated with a clipped PPO surrogate objective. An additional **KL divergence penalty** against a frozen reference model is applied to prevent the policy from drifting too far from its initialization:

- `actor.use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl` (a low-variance KL estimator that reduces gradient noise compared to the standard sample-based KL estimate)
- An optional **entropy coefficient** (`entropy_coeff`) encourages exploration; the R1 variant uses adaptive entropy adjustment to maintain a target entropy.

Training uses FSDP (Fully Sharded Data Parallel) across 8 GPUs with dynamic batching to maximize GPU utilization.

#### Step 7 — Online Curriculum Learning

At regular intervals, successfully scored examples from the current training step are saved back to disk as new Parquet files (`${OUTPUT_DIR}/online_data/`). On subsequent epochs these online examples are merged with the original dataset, providing a growing pool of harder, model-generated verification instances.

#### Step 8 — Evaluation

A validation pass is run every `test_freq` steps on held-out benchmarks (AIME 2024/2025, AMC 2023, MATH-500, OlympiadBench). The evaluation reward (`test_verification_reward`) is a simple binary accuracy: `1.0` if the Yes/No prediction matches the ground-truth label, `0.0` otherwise. Results are logged to Weights & Biases.

#### Full Pipeline Summary

```
Raw data (JSON)
  └─► math_preprocess.py          # Build multi-turn verification prompts → Parquet
        └─► RLHFDataset            # Tokenize & batch prompts
              └─► vLLM rollout (N=8 responses per prompt)
                    └─► compute_score()     # Verification reward (baseline / fix_imbalance)
                          └─► GRPO advantage normalization (group-relative)
                                └─► PPO update + KL penalty
                                      └─► Online data saving → merged into next epoch
                                            └─► Periodic validation on math benchmarks
```

After training completes, the FSDP checkpoints are automatically converted to HuggingFace format for use in the evaluation kit.


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
