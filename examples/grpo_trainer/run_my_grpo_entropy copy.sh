set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export VLLM_USE_V1='1'
export WANDB_PROJECT="verl_grpo_prev_epoch_qwen2_5_1_5b_MATH"
export WANDB_ENTITY="haechan-kaist"  # optional if using teams
export WANDB_MODE="online"  # or "offline", "disabled"
export WANDB_RUN_ID="k2xs8nuz"
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="79f4decc1667e5ef75c38f236c356ee5cc1c764b"
# export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_RESUME='must'


exp_name='grpo-entropy-default-2'

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=1 # d
clip_ratio_high=1 # d
loss_mode="clip_cov"
clip_cov_ratio=0.0002
clip_cov_lb=1.0
clip_cov_ub=5.0

python3 -m recipe.entropy.main_entropy \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/GRPO/data/MATH/train.parquet \
    data.val_files=/workspace/GRPO/data/MATH500/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((1024 + 4096) * 2)) \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio=${clip_cov_ratio} \
    actor_rollout_ref.actor.policy_loss.clip_cov_lb=${clip_cov_lb} \
    actor_rollout_ref.actor.policy_loss.clip_cov_ub=${clip_cov_ub} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(((1024 + 4096) * 8)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_prev_epoch_qwen2_5_1_5b_MATH' \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir="/workspace/GRPO/models/$exp_name" \
    trainer.test_freq=20 \
    trainer.total_epochs=6 \
    trainer.val_before_train=False $@

python3 ../send_msg.py

# trainer.rollout_data_dir='/workspace/GRPO/models/verl_grpohist_qwen2_5_1_5b_MATH/rollout' \