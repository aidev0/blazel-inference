# Blazel Inference

LLM inference service for Blazel - supports local development with Ollama and production deployment with vLLM on GPU.

## Architecture

- **Local**: Ollama with quantized models for fast iteration
- **Production**: vLLM on NVIDIA L4 GPU with Llama 3.1 8B Instruct

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status and backend info |
| `/health` | GET | Health check for load balancers |
| `/generate` | POST | Generate LinkedIn post content |
| `/generate-stream` | POST | Streaming generation (Ollama only) |
| `/adapters` | GET | List active LoRA adapters |
| `/reload-adapter` | POST | Register customer LoRA adapter |

## Local Development

```bash
# Start Ollama
ollama serve

# Pull model
ollama pull llama3.1:8b

# Run service
ENV=local python -m uvicorn app.main:app --reload --port 8001
```

## Production

Deployed on GCP Compute Engine with:
- NVIDIA L4 GPU (24GB VRAM)
- vLLM serving Llama 3.1 8B at 16-bit precision
- systemd service management
- Auto-deploy via GitHub Actions on push to `main`

## API Example

```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a post about AI in healthcare", "max_tokens": 300}'
```
