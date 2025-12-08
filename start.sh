#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .env 2>/dev/null || true

# Set up Python environment for user-installed packages
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Start vLLM server with LoRA support in background
echo "Starting vLLM with LoRA support..."
HF_TOKEN=${HF_TOKEN} python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct} \
    --port 8080 \
    --host 0.0.0.0 \
    --enable-lora \
    --max-lora-rank 64 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 &

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM is ready"

# Start FastAPI inference service
cd "$SCRIPT_DIR"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001
