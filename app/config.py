import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "local")

# Local config (Ollama - quantized for CPU)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Production config (vLLM on GPU - full 16-bit)
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# MongoDB config for fetching active adapters
MONGODB_URL = os.getenv("MONGODB_URL", "")

# GCS bucket for downloading adapters
GCS_BUCKET = os.getenv("GCS_BUCKET", "blazel-adapters")

# Hugging Face token for model downloads
HF_TOKEN = os.getenv("HF_TOKEN", "")
