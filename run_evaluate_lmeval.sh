#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# NCCL sane defaults for single-node multi-GPU on AWS
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1 # This should be needed for vLLM version 0.11.0
# export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# export NCCL_SOCKET_IFNAME=lo


export NCCL_NVLS_ENABLE=0
export VLLM_USE_SYMMETRIC_MEMORY=1
unset NCCL_SOCKET_IFNAME

# Suppress Gloo verbose logs
export GLOO_LOG_LEVEL=ERROR

TP_SIZE=8

# mkdir /dev/shm
# sudo chmod 777 /dev/shm 


TASKS=(
    "bbh_boxed"
    "mmlu_pro_boxed"
    "gpqa_diamond_boxed"
)

MODELS=(
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-again/global_step_405"
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_75/global_step_365"
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_5/global_step_310"
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_25/global_step_450"
    # "qwen3-grpo-paper-dapo17k-batch128-cliph0_2-clipl0_28-clipc3-nokl-lr1e-6-again/global_step_420"
    #########################################################
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6/global_step_315"
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_75/global_step_540"
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_5/global_step_480"
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_25/global_step_465"
    "qwen3-8b-grpo-paper-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6-again/global_step_390"
)


cd /wbl/post-training/lm-evaluation-harness
eval "$(conda shell.bash hook)"
conda activate lm-eval-2


for MODEL in "${MODELS[@]}"; do
    # Set MAX_GEN_TOKS based on model name
    if [[ "$MODEL" == qwen3-8b-* ]]; then
        MAX_GEN_TOKS=8192
    else
        MAX_GEN_TOKS=4096
    fi

    for TASK in "${TASKS[@]}"; do
        HOME_PATH="/wbl/post-training/haechan_workspace/MLILAB-GRPO"
        MODEL_PATH="${HOME_PATH}/models/${MODEL}/actor/huggingface"
        RESULT_PATH="${HOME_PATH}/results/${MODEL}/${TASK}"
        
        echo "*** Evaluate ${MODEL} on ${TASK} with MAX_GEN_TOKS=${MAX_GEN_TOKS} ***"
        lm_eval \
          --model=vllm \
          --tasks=${TASK} \
          --seed=42 \
          --batch_size=auto \
          --model_args=pretrained=${MODEL_PATH},tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=0.9,max_length=$((${MAX_GEN_TOKS} + 4096)),max_gen_toks=${MAX_GEN_TOKS},trust_remote_code=True,disable_cascade_attn=True \
          --gen_kwargs max_gen_toks=${MAX_GEN_TOKS},do_sample=True,temperature=0.6,presence_penalty=0.0,top_p=0.95 \
          --confirm_run_unsafe_code \
          --log_samples \
          --apply_chat_template \
          --output_path ${RESULT_PATH}
    done
done

conda deactivate