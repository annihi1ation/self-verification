WORKING_ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
WORKING_DIR="${WORKING_ROOT_DIR}/verl-train"
cd $WORKING_DIR

unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE
export WANDB_API_KEY="your-wandb-api-key"

DATA_NAME=deepscaler
TRAIN_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/qwen-math/deepscaler/train.parquet)]"
TEST_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/aime24x10.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/aime25x10.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/amc23.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/math500.parquet),$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/olympiadbench.parquet)]"

MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

REWARD_FN_PATH=${WORKING_ROOT_DIR}/verl-train/verl/utils/reward_score/math_verification.py

VERIFICATION_REWARD_TYPE=baseline
AUXILIARY_REWARDS=none

EXP_NOTE=verifier-16k-${VERIFICATION_REWARD_TYPE}
GPU_NUMS=8

PROJECT_NAME=verifier-verl
EXPERIMENT_NAME=${EXP_NOTE}-${MODEL_NAME}-${DATA_NAME}-NODE${WORLD_SIZE}
OUTPUT_DIR=${WORKING_ROOT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
mkdir -p ${OUTPUT_DIR}

PROMPT_LENGTH=2048
# RESPONSE_LENGTH=6144
RESPONSE_LENGTH=14336
NUM_BATCHED_TOKENS=$((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}))
MAX_TOKEN_PER_GPU=$(((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}) * 1))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${TEST_DATA_PATH} \
    data.train_batch_size=128 \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.005 \
    actor_rollout_ref.actor.use_adaptive_entropy_adjustment=True \
    actor_rollout_ref.actor.target_entropy=0.2 \
    actor_rollout_ref.actor.entropy_coeff_delta=0.0001 \
    actor_rollout_ref.actor.max_entropy_coeff=0.005 \
    actor_rollout_ref.actor.min_entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=${NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=naive \
    custom_reward_function.path=${REWARD_FN_PATH} \
    reward_config.verification_reward_type=${VERIFICATION_REWARD_TYPE} \
    reward_config.auxiliary_rewards=${AUXILIARY_REWARDS} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${GPU_NUMS} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.rejection_sample=True \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=2000 \
    trainer.save_online_data.enabled=True \
    trainer.save_online_data.train_data_size=40000 \
    trainer.save_online_data.val_data_size=200 \
    trainer.save_online_data.online_save_freq=60 \
    2>&1 | tee ${OUTPUT_DIR}/train.log

cd ${WORKING_ROOT_DIR}
python scripts/model_utils/convert_final_ckpt.py ${OUTPUT_DIR}
