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
    "OlympiadBench"
)

MODELS=(
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6/global_step_315"
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_75/global_step_540"
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_5/global_step_480"
    "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_25/global_step_465"
    "qwen3-8b-grpo-paper-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6-again/global_step_390"
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-again/global_step_405"
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_75/global_step_365"
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_5/global_step_310"
    # "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_25/global_step_450"
    # "qwen3-grpo-paper-dapo17k-batch128-cliph0_2-clipl0_28-clipc3-nokl-lr1e-6-again/global_step_420"
)



for MODEL in "${MODELS[@]}"; do
    for DATA_NAME in "${DATA_NAMES[@]}"; do
        echo "[evaluate.sh] Evaluating ${DATA_NAME} at model ${MODEL}"
        HOME_PATH="/workspace/GRPO"
        ACTOR_PATH="${HOME_PATH}/models/${MODEL}/actor"
        MODEL_PATH="${ACTOR_PATH}/huggingface"
        DATA_PATH="${HOME_PATH}/data/${DATA_NAME}/test.parquet"
        RESULT_PATH="${HOME_PATH}/results/${MODEL}/${DATA_NAME}"

        # Convert FSDP checkpoint to safetensors if not already done
        if [ ! -f "${MODEL_PATH}/model.safetensors" ]; then
            echo "[evaluate.sh] model.safetensors not found, converting FSDP checkpoint..."
            python convert_model.py --actor_path "${ACTOR_PATH}"
        fi

        python evaluate.py \
            --model_path ${MODEL_PATH} \
            --data_path ${DATA_PATH} \
            --output_dir ${RESULT_PATH} \
            --tensor_parallel_size 8 \
            --n_samples 8 \
            --temperature 0.6 \
            --top_p 0.95 \
            --max_tokens 8192

    done
done
