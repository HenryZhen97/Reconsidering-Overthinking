#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
COMPRESS=true
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
EXP_NAME=truncate

EXTERNAL_REDUNDANCY=false
INTERNAL_REDUNDANCY=true

if [ "$EXTERNAL_REDUNDANCY" = true ]; then
    EXP_NAME=${EXP_NAME}-redundancy
fi

if [ "$INTERNAL_REDUNDANCY" = true ]; then
    EXP_NAME=${EXP_NAME}-gain
fi

NNODES=8
PROJECT_NAME=compress_rl_7B
BATCH_SIZE=256
N=8
RESPONSE_LENGTH=8192
VAL_LENGTH=16384

if [[ "$MODEL_PATH" == *huggingface* ]]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
elif [[ "$MODEL_PATH" == *compress_rl* ]]; then

    STEP=$(basename "$MODEL_PATH" | grep -oP 'global_step_\K[0-9]+')
    

    TRUNC_PART=$(echo "$MODEL_PATH" | grep -oP 'compress_rl-[^/]*' | grep -oP 'truncate.*')


    MODEL_NAME="${TRUNC_PART}_step_${STEP}"
else

    echo "❌ Unsupported MODEL_PATH: $MODEL_PATH"
    exit 1
fi

EXP_NAME=${PROJECT_NAME}-${MODEL_NAME}-${EXP_NAME}-n${N}-bs${BATCH_SIZE}-L$((${RESPONSE_LENGTH} / 1024))k
LOG_FILE="./logs/${EXP_NAME}.log"

# Train over 4 nodes, 8 A800-40GB GPUs per node.
nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/rllm/data/deepscaler_train.parquet \
    data.val_files=[$HOME/rllm/data/aime.parquet,$HOME/rllm/data/math.parquet] \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=$VAL_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9000 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.compress=$COMPRESS \
    actor_rollout_ref.rollout.compute_reward=true \
    +actor_rollout_ref.rollout.train_max_tokens=$RESPONSE_LENGTH \
    +actor_rollout_ref.rollout.val_max_tokens=$VAL_LENGTH \
    +actor_rollout_ref.rollout.use_external_redundancy=$EXTERNAL_REDUNDANCY \
    +actor_rollout_ref.rollout.use_internal_redundancy=$INTERNAL_REDUNDANCY \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.n_val=8 \
    +actor_rollout_ref.rollout.n_val_aime=32 \
    +actor_rollout_ref.rollout.n_val_math=1 \
    +actor_rollout_ref.rollout.n_val_gsm8k=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.mask_truncated_samples=true \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 "${@:1}" \
    2>&1 > $LOG_FILE &
