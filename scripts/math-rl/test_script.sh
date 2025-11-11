WORKING_ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
echo $WORKING_ROOT_DIR
TRAIN_FILES=${WORKING_ROOT_DIR}/data-backup/qwen-math/math-level3to5/train-50.parquet
VAL_FILES="[$(echo ${WORKING_ROOT_DIR}/data/eval-data-no-system-prompt/aime24.parquet)]"

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=${TRAIN_FILES} \
 data.val_files=${VAL_FILES} \
 data.train_batch_size=32 \
 data.val_batch_size=1312 \
 data.max_prompt_length=2048 \
 data.max_response_length=2048 \
 actor_rollout_ref.model.path=${WORKING_ROOT_DIR}/models/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size=1 \
 actor_rollout_ref.rollout.name=sglang \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.load_format=dtensor \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=${WORKING_ROOT_DIR}/models/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size=1 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=5 \
 trainer.default_local_dir=${WORKING_ROOT_DIR}/checkpoints/test-cases \
 trainer.name=verifier \
 trainer.save_online_data.enabled=True \
 trainer.logger=\[console\]
 