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

DATA_NAMES=(
    # "AIME2024"
    # "AIME2025"
    # "AMC2023"
    # "AMC2024"
    # "MATH500"
    # "Minerva"
    "OlympiadBench"
)

# 10부터 100까지 10단위씩 반복하며 평가
# STEPS=(
#     $(seq 10 10 1050)
# )
STEPS=(
    250
)


for STEP in "${STEPS[@]}"; do
    for DATA_NAME in "${DATA_NAMES[@]}"; do
        echo "[evaluate.sh] Evaluating ${DATA_NAME} at step ${STEP}"
        HOME_PATH="/workspace"
        MODEL_PATH="${HOME_PATH}/RePO/outputs/qwen3-repo-paper-dapo17k-batch128-cliph0_28-clipl0_2-nokl-lr1e-6-20251221_005737/checkpoint-${STEP}"
        DATA_PATH="${HOME_PATH}/MLILAB-GRPO/data/${DATA_NAME}/test.parquet"
        RESULT_PATH="${HOME_PATH}/RePO/results/qwen3-repo-paper-dapo17k-batch128-cliph0_28-clipl0_2-nokl-lr1e-6-20251221_005737/checkpoint-${STEP}/${DATA_NAME}"

        python evaluate.py \
            --model_path ${MODEL_PATH} \
            --data_path ${DATA_PATH} \
            --output_dir ${RESULT_PATH} \
            --tensor_parallel_size 8 \
            --n_samples 8 \
            --temperature 0.6 \
            --top_p 0.95 \
            --max_tokens 4096
    done
done
