set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_V1='1'
export WANDB_PROJECT="GRPO"
export WANDB_ENTITY="haechan-kaist"  # optional if using teams
export WANDB_MODE="online"  # or "offline", "disabled"
export WANDB_RUN_ID="khb6gxwj"
export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_RESUME='must'

experiment_name="qwen3-grpo-clip-1_0-clipc-10-nokl-lr-0.5e-6"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/GRPO/data/MATH/train.parquet \
    data.val_files=/workspace/GRPO/data/MATH500/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=0.5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((1024 + 4096) * 2)) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=1.0 \
    actor_rollout_ref.actor.clip_ratio_high=1.0 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(((1024 + 4096) * 8)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_prev_epoch_qwen2_5_1_5b_MATH' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir="/workspace/GRPO/models/$experiment_name" \
    trainer.test_freq=20 \
    trainer.total_epochs=12 \
    trainer.val_before_train=False $@

python3 ../send_msg.py

# trainer.rollout_data_dir='/workspace/GRPO/models/verl_grpohist_qwen2_5_1_5b_MATH/rollout' \