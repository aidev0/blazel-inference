# Blazel Inference

LLM inference service with dynamic LoRA adapter loading. Supports Ollama (local development) and vLLM (production with GPU).

## Architecture

```
┌──────────────────┐      ┌─────────────────┐
│  FastAPI Server  │─────▶│  vLLM (prod)    │
│  (Port 8001)     │      │  (Port 8080)    │
└────────┬─────────┘      └─────────────────┘
         │
         ▼
┌──────────────────┐      ┌─────────────────┐
│  MongoDB         │      │  GCS Bucket     │
│  (adapter lookup)│      │  (adapter files)│
└──────────────────┘      └─────────────────┘
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service status and backend info |
| GET | `/health` | Health check (verifies vLLM/Ollama connection) |
| POST | `/generate` | Generate LinkedIn post |
| POST | `/generate-stream` | Streaming generation (Ollama only) |

### POST /generate

Request:
```json
{
  "prompt": "Write a post about AI in healthcare",
  "customer_id": "cust_123",
  "max_tokens": 500,
  "temperature": 0.7
}
```

Response:
```json
{
  "text": "Generated post content...",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "customer_id": "cust_123",
  "backend": "vllm"
}
```

## LoRA Adapter Flow

1. Request comes in with `customer_id`
2. Service queries MongoDB for active adapter
3. If adapter found, downloads from GCS (if not cached locally)
4. Dynamically loads adapter into vLLM via `/v1/load_lora_adapter`
5. Generates using the personalized adapter

## Environment Variables

```bash
ENV=production              # "production" uses vLLM, else Ollama
VLLM_URL=http://localhost:8080
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
MONGODB_URL=mongodb+srv://...
HF_TOKEN=hf_...             # For downloading Llama model
```

## GCP VM Setup (Production)

### Prerequisites
- GCP project with Compute Engine API enabled
- T4 GPU VM (at minimum)
- Service account with GCS access

### VM Specifications
```
Machine Type: n1-standard-4 (or higher)
GPU: NVIDIA T4
Disk: 100GB SSD
Zone: us-west1-a
Image: Deep Learning VM with PyTorch
```

### Installation Steps

1. **SSH into the VM:**
```bash
gcloud compute ssh jacobrafati@blazel-inference --zone=us-west1-a --project=blazel-prod
```

2. **Clone the repository:**
```bash
cd ~
git clone https://github.com/aidev0/blazel-inference.git
cd blazel-inference
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install vllm  # Requires CUDA
```

4. **Create .env file:**
```bash
cat > .env << 'EOF'
ENV=production
VLLM_URL=http://localhost:8080
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=hf_your_token_here
MONGODB_URL=mongodb+srv://...
EOF
```

5. **Set up vLLM as a systemd service:**
```bash
sudo tee /etc/systemd/system/vllm.service > /dev/null << 'EOF'
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
Type=simple
User=jacobrafati
WorkingDirectory=/home/jacobrafati/blazel-inference
Environment="HF_TOKEN=hf_your_token_here"
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8080 \
    --enable-lora \
    --max-lora-rank 64 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

6. **Set up FastAPI service:**
```bash
sudo tee /etc/systemd/system/blazel-inference.service > /dev/null << 'EOF'
[Unit]
Description=Blazel Inference FastAPI
After=network.target vllm.service

[Service]
Type=simple
User=jacobrafati
WorkingDirectory=/home/jacobrafati/blazel-inference
ExecStart=/usr/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable blazel-inference
sudo systemctl start blazel-inference
```

### Firewall Rules

Open port 8001 for the inference API:
```bash
gcloud compute firewall-rules create allow-inference \
    --allow tcp:8001 \
    --target-tags=inference-server \
    --project=blazel-prod
```

## Common Issues & Solutions

### Issue: vLLM OOM (Out of Memory)
**Symptom:** vLLM crashes during model load
**Solution:** Add `--max-model-len 8192` to limit context window

### Issue: LoRA adapter not loading
**Symptom:** "Failed to load adapter" in logs
**Solution:**
- Verify adapter files exist: `ls /home/jacobrafati/adapters/{customer_id}/`
- Check adapter_config.json is present
- Ensure vLLM started with `--enable-lora --max-lora-rank 64`

### Issue: GCS download fails
**Symptom:** "gsutil command not found" or permission denied
**Solution:**
```bash
# Install gsutil
sudo snap install google-cloud-sdk --classic

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### Issue: MongoDB connection fails
**Symptom:** "MongoDB connection failed" in logs
**Solution:** Verify MONGODB_URL in .env and whitelist VM IP in MongoDB Atlas

## Monitoring

Check service status:
```bash
sudo systemctl status vllm
sudo systemctl status blazel-inference
```

View logs:
```bash
journalctl -u vllm -f
journalctl -u blazel-inference -f
```

Check GPU usage:
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## Local Development

For local development without GPU, use Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama model
ollama pull llama3.1:8b

# Run with ENV=development
ENV=development python -m uvicorn app.main:app --reload --port 8001
```

## Deployment

### Manual SSH Deploy
```bash
ssh -i ~/.ssh/blazel-inference-deploy jacobrafati@35.229.82.124
cd ~/blazel-inference
git pull
sudo systemctl restart blazel-inference
```

### Current Production
- **IP:** 35.229.82.124
- **Port:** 8001
- **URL:** http://35.229.82.124:8001
