#!/bin/bash

# Start vLLM server in background
python3 -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --port 8080 \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 &

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM is ready"

# Start FastAPI inference service
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001
