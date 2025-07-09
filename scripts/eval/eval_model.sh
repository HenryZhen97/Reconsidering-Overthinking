set -x

CODE_DIR=<rllm_root>
cd $CODE_DIR

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

if [[ "$MODEL_PATH" == *huggingface* ]]; then
    DIR_NAME=$(basename "$MODEL_PATH")
else
    DIR_NAME=$(basename "$(dirname "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH" | grep -o 'step_[0-9]\+')")
fi

# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime" "math" "gsm8k")
declare -A N_PASSES_MAP=(
    [aime]=64
    [math]=4
    [gsm8k]=4
)

declare -A TP_MAP=(
    [aime]=4
    [math]=4
    [gsm8k]=4
)

GPU_USAGE=0.7
MAX_LENGTH=16384  # Default max response length
DIR_NAME=$DIR_NAME-L$((${MAX_LENGTH} / 1024))k
OUTPUT_DIR="$CODE_DIR/eval_output/$DIR_NAME"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n)
            N_PASSES="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory> --n <number_of_passes> --max-length <max_response_length> --tp <tensor_parallel_size>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of Passes: ${N_PASSES}"
echo "Max Response Length: ${MAX_LENGTH}"
echo "Tensor Parallel Size: ${TP_SIZE}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    N_PASSES=${N_PASSES_MAP[$DATA_TYPE]}
    TP_SIZE=${TP_MAP[$DATA_TYPE]}
    echo "Number of Passes: ${N_PASSES}"
    echo "TP: ${TP_SIZE}"

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$HOME/rllm/data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=${N_PASSES} \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=${MAX_LENGTH} \
        +rollout.train_max_tokens=${MAX_LENGTH} \
        +rollout.val_max_tokens=${MAX_LENGTH} \
        rollout.top_k=-1 \
        rollout.top_p=1 \
        rollout.gpu_memory_utilization=$GPU_USAGE \
        rollout.tensor_model_parallel_size=${TP_SIZE}
done