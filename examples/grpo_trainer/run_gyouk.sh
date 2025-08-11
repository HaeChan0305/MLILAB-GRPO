#!/bin/bash
set -x

project_name='LengthRL'
exp_name='MATH-1.7b-4k-truncation-4x3090'
MODEL_PATH="Qwen/Qwen3-1.7B"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_FILENAME="${SCRIPT_DIR}/Qwen3-1.7B-MATH-truncation.txt"

math_train_path="${SCRIPT_DIR}/data/MATH-train-offline-filtered.parquet"
math_test_path="${SCRIPT_DIR}/data/MATH-test.parquet"
TRAIN_FILES="['$math_train_path']"
TEST_FILES="['$math_test_path']"

rollout_data_dir="${SCRIPT_DIR}/rollout_save_folder4/"

MAX_PROMPT_LENGTH=$((1024 * 1))
MAX_RESPONSE_LENGTH=$((1024 * 4)) # Overlong Buffer 없이 쌩 4k
GEN_PROMPT_BSZ=32 # 3 x TRAIN_PROMPT_BSZ
TRAIN_PROMPT_BSZ=32
N_RESP_PER_PROMPT=8
TRAIN_PROMPT_MINI_BSZ=8
USE_DYNAMIC_BSZ=True
PPO_PER_GPU_MAX_TOKEN_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

LR=5e-7
WEIGHT_DECAY=0.0 # Turn off
USE_KL_LOSS=True
KL_LOSS_COEF=0.001
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28 # DAPO

OFFLOAD=True # All-in-one
ULYSSES_TP=1 # Turn off
GEN_TP=4 # Turn on

TEMPERATURE=0.6
TOP_P=1.0
VAL_TEMPERATURE=0.6
VAL_TOP_P=0.95

python3 -m recipe.gyouk.custom_main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric="acc" \
    algorithm.filter_groups.max_num_gen_batches=10 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.gen_batch_size=${GEN_PROMPT_BSZ} \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ} \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_PER_GPU_MAX_TOKEN_LENGTH} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_TP} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}" > >(tee -a ${LOG_FILENAME})