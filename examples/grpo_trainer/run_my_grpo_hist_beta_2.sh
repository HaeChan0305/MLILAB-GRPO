set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_V1='1'
export WANDB_PROJECT="GRPO"
export WANDB_ENTITY="haechan-kaist"  # optional if using teams
export WANDB_MODE="online"  # or "offline", "disabled"
# export WANDB_RUN_ID="q21nr2kw"
export HYDRA_FULL_ERROR=1
# export WANDB_RESUME='must'

experiment_name="qwen3-grpohistbeta-test"
save_path="/workspace/GRPO/models/$experiment_name"
freq=20
rollout=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpohistbeta \
    data.train_files=/workspace/GRPO/data/MATH/train_example_7495.parquet \
    data.val_files=/workspace/GRPO/data/MATH500/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=0.5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.n=$rollout \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(((1024 + 4096) * 8)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.history.path=$save_path \
    algorithm.history.save_freq=$freq \
    algorithm.history.rollout_n=$rollout \
    algorithm.history.discount=1.0 \
    algorithm.history.alpha_init=1.0 \
    algorithm.history.beta_init=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_prev_epoch_qwen2_5_1_5b_MATH' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=$freq \
    trainer.default_local_dir=$save_path \
    trainer.test_freq=$freq \
    trainer.total_epochs=6 \
    trainer.val_before_train=False $@

python3 ../send_msg.py

# trainer.rollout_data_dir='/workspace/GRPO/models/verl_grpohist_qwen2_5_1_5b_MATH/rollout' \