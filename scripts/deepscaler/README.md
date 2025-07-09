
# CoT Compress Scripts

## Multi-Node Training
To run, follow these steps:
1. On the head node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Start Ray head node
ray start --head
```

2. On each worker node:
```bash
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Connect to head node (replace with your head node's address)
ray start --address=[RAY_ADDRESS]
```

3. On each worker node, run embedding service:
```bash
# Start embedding service
./rllm/rewards/compress_utils/start_app.sh [WORKER_ID]
```

4. Finally, on the head node, run the training script:
```bash
# Run 16K or 24K context length training
./scripts/train/deepscaler_[1.5b|7b]_compress.sh
```
