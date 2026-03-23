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

---

# 基于大语言模型验证器的自我验证（中文说明）

本项目是论文《激励大语言模型自我验证其答案》的开源实现。

## 安装

训练代码基于 verl 框架。请先参考 verl 的安装指南：https://verl.readthedocs.io/en/latest/start/install.html。推荐安装 `vllm==0.8.2` 以获得最佳复现效果。

安装 verl 后，还需要安装 math-verify：

```bash
pip install math-verify
```

## 自我验证模型的 GRPO 训练

训练代码位于 `verl-train` 目录下。强化学习脚本分别为：
- `scripts/math-rl/grpo-qwen-verifier.sh`：用于训练 Self-Verification-Qwen-7B 模型
- `scripts/math-rl/grpo-r1-verifier.sh`：用于训练 Self-Verification-R1-1.5B 模型

如需在多节点上运行，还需启动 ray 服务器，请参考 `scripts/ray_start_script.sh`。

训练前，请确认以下环境变量和配置已在对应的训练脚本中正确设置：

1. `WORKING_DIR` 设置为当前目录（与 README 文件同级）。
2. `WANDB_API_KEY` 设置为你的 wandb API 密钥。
3. `MODEL_PATH` 根据已下载的模型路径正确设置。

然后使用以下命令运行训练脚本：

```bash
bash scripts/math-rl/grpo-qwen-verifier.sh
```

建议至少使用 8 块 GPU 以保证训练效率。

### 训练流程

GRPO 自我验证模型的训练管道包含以下阶段：

#### 第一步 — 数据准备（`scripts/math-rl/math_preprocess.py`）

将原始数学题数据集（JSON/JSONL 格式）转换为 Parquet 文件。每个训练样本被格式化为**多轮对话**，模型同时看到题目和候选解答，然后被要求对其进行验证：

```
第一轮（用户）：    <数学题目>
第二轮（助手）：    <候选解答>
第三轮（用户）：    "请逐步验证上述解答。在完成验证后给出最终评级时，
                   请使用如下格式：'Is the answer correct (Yes/No)? X'，
                   其中 X 为 Yes 或 No。"
```

每个样本的元数据中包含 `ground_truth` 字段，编码格式为 `"<label>|<correct_ratio>"`：
- `label`：Python 字符串 `'True'`/`'False'`（表示候选解答是否正确，在奖励计算时通过 `eval()` 转换为布尔值）
- `correct_ratio`：该组中正确解答所占的比例（用于不平衡感知的奖励塑形）

#### 第二步 — 数据加载

`RLHFDataset` 使用模型的对话模板对多轮提示进行分词，过滤掉超过 `max_prompt_length` 的提示，并按批次输出 `(input_ids, attention_mask, reward_model_metadata)`。

#### 第三步 — Rollout / 生成

每个训练步骤中，当前 Actor 策略使用 vLLM 为每个提示生成 **N = 8 个回复**（可通过 `actor_rollout_ref.rollout.n` 配置）。每个回复是一段思维链式的验证过程，最终以判断 `"Is the answer correct (Yes/No)? Yes"` 或 `"...? No"` 作为结尾。

#### 第四步 — 奖励评分（`verl/utils/reward_score/math_verification.py`）

奖励函数（`compute_score`）将每个生成的回复与真实标签进行对比。通过环境变量 `VERIFICATION_REWARD_TYPE` 可选择两种奖励类型：

| 类型 | 公式 |
|---|---|
| `baseline` | 若预测与标签一致，`reward = 1.0`；否则为 `0.0` |
| `fix_imbalance` | 预测正确时：`reward = 2 × [float(pred) × (1 − correct_ratio) + (1 − float(pred)) × correct_ratio]`；否则为 `0.0`。其中 `float(True) = 1.0`，`float(False) = 0.0`，因此正确预测"是"时权重为 `2 × (1 − correct_ratio)`，正确预测"否"时权重为 `2 × correct_ratio`。 |

`fix_imbalance` 类型通过上调难以预测样本的权重来应对验证标签中的类别不平衡问题。

还可通过 `AUXILIARY_REWARDS` 添加可选的**辅助惩罚**：
- `non_short_response`：若回复少于 40 个字符，扣 `−0.5` 分。
- `no_code`：若回复包含 Python 代码块，扣 `−0.5` 分。

#### 第五步 — GRPO 优势估计（`verl/trainer/ppo/core_algos.py`）

回复按所对应的提示进行分组，组内对标量奖励进行归一化：

```
advantage_i = (score_i − mean(group_scores)) / (std(group_scores) + ε)
```

这是 GRPO（组相对策略优化）的核心机制：无需独立的 Critic 网络，优势值完全由组内各回复的相对质量决定。

#### 第六步 — 策略更新

Actor 使用带截断的 PPO 代理目标进行更新。同时施加一个针对冻结参考模型的 **KL 散度惩罚**，以防止策略偏离初始化过远：

- `actor.use_kl_loss=True`，`kl_loss_coef=0.001`，`kl_loss_type=low_var_kl`（低方差 KL 估计器，与标准的基于采样的 KL 估计相比可减少梯度噪声）
- 可选的**熵系数**（`entropy_coeff`）鼓励探索；R1 变体使用自适应熵调整以维持目标熵。

训练使用 FSDP（全分片数据并行）跨 8 块 GPU 运行，并采用动态批大小以最大化 GPU 利用率。

#### 第七步 — 在线课程学习

每隔一定时间间隔，当前训练步骤中评分成功的样本会被保存回磁盘，形成新的 Parquet 文件（保存于 `${OUTPUT_DIR}/online_data/`）。在后续训练轮次中，这些在线样本将与原始数据集合并，持续构建由模型自身生成的、更具挑战性的验证样本池。

#### 第八步 — 评估

每隔 `test_freq` 步，在留出的基准数据集上进行一次验证（AIME 2024/2025、AMC 2023、MATH-500、OlympiadBench）。评估奖励（`test_verification_reward`）为简单的二值准确率：若 Yes/No 预测与真实标签一致则为 `1.0`，否则为 `0.0`。结果记录至 Weights & Biases。

#### 完整流程概览

```
原始数据（JSON）
  └─► math_preprocess.py          # 构建多轮验证对话提示 → Parquet
        └─► RLHFDataset            # 分词并批处理提示
              └─► vLLM rollout（每个提示生成 N=8 个回复）
                    └─► compute_score()     # 验证奖励（baseline / fix_imbalance）
                          └─► GRPO 优势归一化（组相对）
                                └─► PPO 更新 + KL 惩罚
                                      └─► 在线数据保存 → 合并入下一轮训练
                                            └─► 定期在数学基准上进行验证
```

训练完成后，FSDP 检查点会自动转换为 HuggingFace 格式，供评估工具包使用。


## 自我验证模型的测试时扩展

测试时扩展代码位于 `evaluation-kit` 目录下。

```bash
cd evaluation-kit
```

评估配置文件位于 `evaluation-kit/configs` 目录中。推理阶段使用 sglang 库，请参考 sglang 文档进行安装：https://docs.sglang.ai/start/install.html。常用安装命令如下：

```bash
pip install "sglang[all]>=0.4.6.post4"
```

此外，还需要将训练好的模型转换为 HuggingFace 格式。如果训练脚本已成功运行，将会自动将 FSDP 检查点转换为 HuggingFace 格式。也可以手动执行以下命令进行转换：

```bash
python scripts/model_utils/convert_final_ckpt.py ${OUTPUT_DIR}
```

其中 `${OUTPUT_DIR}` 为训练脚本的输出目录（由 `OUTPUT_DIR` 环境变量指定）。

使用自我验证进行测试时扩展，可运行以下命令：

```bash
python run_self_verify.py configs/Qwen-rl-models/qwen-rl-verifier.yaml --model_path=path-to-hf-model --model_name=self-verification --dataset_path=data/math500.jsonl --n_samples=16 
```

configs 目录中还提供了其他测试时扩展基线。例如，将配置文件替换为 `configs/Qwen-rl-models/qwen-rl-bon.yaml` 可运行 Best-of-N 基线：

```bash
python run_search.py configs/Qwen-rl-models/qwen-rl-bon.yaml --model_path=path-to-hf-model --model_name=self-verification --dataset_path=data/math500.jsonl --n_samples=16 
```

也可以将 `dataset_path` 替换为 `data` 目录下的其他数据集。
