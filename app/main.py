from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
import subprocess
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import ENV, OLLAMA_URL, OLLAMA_MODEL, VLLM_URL, VLLM_MODEL, MONGODB_URL

app = FastAPI(
    title="Blazel Inference",
    description="LLM Inference Service - Ollama (local) or vLLM (prod)",
    version="0.1.0"
)

# Track which LoRA adapters are loaded in vLLM
loaded_adapters: set = set()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection for fetching active adapters
mongo_client = None
db = None

@app.on_event("startup")
async def startup_db():
    global mongo_client, db
    if MONGODB_URL:
        try:
            mongo_client = AsyncIOMotorClient(MONGODB_URL)
            db = mongo_client.blazel
            print(f"[INFERENCE] Connected to MongoDB")
        except Exception as e:
            print(f"[INFERENCE] MongoDB connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_db():
    global mongo_client
    if mongo_client:
        mongo_client.close()


async def get_active_adapter(customer_id: str) -> Optional[dict]:
    """Fetch the active adapter for a customer from MongoDB"""
    if db is None:
        return None
    try:
        adapter = await db.adapters.find_one({
            "customer_id": customer_id,
            "is_active": True
        })
        return adapter
    except Exception as e:
        print(f"[INFERENCE] Error fetching adapter: {e}")
        return None


# Local adapters directory on the inference VM
ADAPTERS_DIR = "/home/jacobrafati/adapters"


def download_adapter_from_gcs(gcs_path: str, local_path: str) -> bool:
    """Download adapter from GCS to local path if not already present"""
    # Check if adapter already exists locally
    config_file = Path(local_path) / "adapter_config.json"
    if config_file.exists():
        print(f"[INFERENCE] Adapter already exists at {local_path}")
        return True

    # Create directory
    Path(local_path).mkdir(parents=True, exist_ok=True)

    try:
        print(f"[INFERENCE] Downloading adapter from {gcs_path} to {local_path}")
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_path}/*", local_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"[INFERENCE] Adapter downloaded successfully")
            return True
        else:
            print(f"[INFERENCE] GCS download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[INFERENCE] Error downloading adapter: {e}")
        return False


def get_adapter_local_path(adapter: dict) -> Optional[str]:
    """Get or download the adapter, return local path from adapter record"""
    local_path = adapter.get("local_path")
    gcs_url = adapter.get("gcs_url", "")

    # Check if adapter already exists locally
    if local_path and Path(local_path).exists() and (Path(local_path) / "adapter_config.json").exists():
        return local_path

    # Download from GCS if we have the URL
    if gcs_url.startswith("gs://") and local_path:
        if download_adapter_from_gcs(gcs_url, local_path):
            return local_path

    print(f"[INFERENCE] No adapter found at {local_path}")
    return None


async def load_lora_adapter(adapter_name: str, adapter_path: str) -> bool:
    """Dynamically load a LoRA adapter into vLLM using the /v1/load_lora_adapter endpoint"""
    global loaded_adapters

    # Skip if already loaded
    if adapter_name in loaded_adapters:
        print(f"[INFERENCE] Adapter {adapter_name} already loaded")
        return True

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{VLLM_URL}/v1/load_lora_adapter",
                json={
                    "lora_name": adapter_name,
                    "lora_path": adapter_path
                }
            )
            if response.status_code == 200:
                loaded_adapters.add(adapter_name)
                print(f"[INFERENCE] Loaded LoRA adapter: {adapter_name} from {adapter_path}")
                return True
            else:
                print(f"[INFERENCE] Failed to load adapter: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        print(f"[INFERENCE] Error loading adapter: {e}")
        return False


class GenerateRequest(BaseModel):
    prompt: str
    customer_id: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    model: str
    customer_id: Optional[str] = None
    backend: str

SYSTEM_PROMPT = """You are an expert LinkedIn content writer. Write engaging, professional posts that:
- Start with a strong hook
- Use short paragraphs
- Include a clear call-to-action
- Are authentic and not overly salesy
- Are between 150-300 words

Write only the post content, no explanations."""


def get_backend_info():
    if ENV == "production":
        return {"backend": "vllm", "url": VLLM_URL, "model": VLLM_MODEL}
    return {"backend": "ollama", "url": OLLAMA_URL, "model": OLLAMA_MODEL}


@app.get("/")
async def root():
    info = get_backend_info()
    return {
        "status": "ok",
        "service": "blazel-inference",
        "env": ENV,
        "backend": info["backend"],
        "model": info["model"]
    }


@app.get("/health")
async def health():
    info = get_backend_info()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if info["backend"] == "ollama":
                response = await client.get(f"{info['url']}/api/tags")
            else:
                response = await client.get(f"{info['url']}/health")

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "backend": info["backend"],
                    "model": info["model"]
                }
    except Exception as e:
        return {"status": "unhealthy", "backend": info["backend"], "error": str(e)}
    return {"status": "unhealthy", "backend": info["backend"]}


async def generate_ollama(prompt: str, request: GenerateRequest) -> str:
    """Generate using Ollama (local, quantized)"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json().get("response", "")


async def generate_vllm(prompt: str, request: GenerateRequest, adapter_path: Optional[str] = None, customer_id: Optional[str] = None) -> str:
    """Generate using vLLM (production, full 16-bit on GPU)

    If adapter_path is provided, dynamically loads the LoRA adapter into vLLM
    and uses it for generation.
    """
    model_to_use = VLLM_MODEL

    # If adapter provided, load it dynamically and use its name
    if adapter_path and customer_id:
        adapter_name = f"adapter-{customer_id}"
        loaded = await load_lora_adapter(adapter_name, adapter_path)
        if loaded:
            model_to_use = adapter_name
            print(f"[INFERENCE] Using LoRA adapter: {adapter_name}")
        else:
            print(f"[INFERENCE] Failed to load adapter, falling back to base model")

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "repetition_penalty": 1.1,
            "stop": ["---", "Here are", "What's your"]
        }

        # vLLM uses OpenAI-compatible chat API for Llama 3.1
        response = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    info = get_backend_info()

    # Look up active adapter from MongoDB for this customer
    adapter_path = None
    if request.customer_id and info["backend"] == "vllm":
        adapter = await get_active_adapter(request.customer_id)
        if adapter:
            # Get or download the adapter, returns local path
            adapter_path = get_adapter_local_path(adapter)
            if adapter_path:
                print(f"[INFERENCE] Customer {request.customer_id} using adapter at {adapter_path}")

    try:
        if info["backend"] == "ollama":
            # Ollama uses raw prompt with system prompt prepended
            full_prompt = f"{SYSTEM_PROMPT}\n\n{request.prompt}"
            generated_text = await generate_ollama(full_prompt, request)
        else:
            # vLLM uses chat format - load adapter dynamically if available
            generated_text = await generate_vllm(request.prompt, request, adapter_path=adapter_path, customer_id=request.customer_id)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"{info['backend']} unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return GenerateResponse(
        text=generated_text.strip(),
        model=info["model"],
        customer_id=request.customer_id,
        backend=info["backend"]
    )


@app.post("/generate-stream")
async def generate_stream(request: GenerateRequest):
    """Streaming generation (Ollama only for now)"""
    from fastapi.responses import StreamingResponse

    info = get_backend_info()

    if info["backend"] != "ollama":
        raise HTTPException(status_code=501, detail="Streaming only supported with Ollama backend")

    full_prompt = f"{SYSTEM_PROMPT}\n\n{request.prompt}"

    async def stream_generator():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    }
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield f"data: {json.dumps({'text': data['response']})}\n\n"
                        if data.get("done"):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
