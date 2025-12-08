from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import ENV, OLLAMA_URL, OLLAMA_MODEL, VLLM_URL, VLLM_MODEL, MONGODB_URL, GCS_BUCKET

app = FastAPI(
    title="Blazel Inference",
    description="LLM Inference Service - Ollama (local) or vLLM (prod)",
    version="0.1.0"
)

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
    if not db:
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


# Cache of loaded adapters (adapter_id -> local_path)
active_adapters: dict = {}

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

class ReloadAdapterRequest(BaseModel):
    customer_id: str
    adapter_path: str

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


async def generate_vllm(prompt: str, request: GenerateRequest, adapter_name: Optional[str] = None) -> str:
    """Generate using vLLM (production, full 16-bit on GPU)

    If adapter_name is provided and vLLM was started with --enable-lora,
    it will use the specified LoRA adapter for generation.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Build request payload
        payload = {
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "repetition_penalty": 1.1,
            "stop": ["---", "Here are", "What's your"]
        }

        # If adapter is available, tell vLLM to use it
        # This requires vLLM to be started with --enable-lora and the adapter registered
        if adapter_name:
            payload["model"] = adapter_name  # vLLM uses adapter name as model when LoRA is enabled
            print(f"[INFERENCE] Requesting generation with adapter: {adapter_name}")

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
    adapter = None
    adapter_name = None
    if request.customer_id and info["backend"] == "vllm":
        adapter = await get_active_adapter(request.customer_id)
        if adapter and adapter.get("gcs_path"):
            adapter_name = f"adapter-{request.customer_id}"
            print(f"[INFERENCE] Using adapter {adapter_name} for customer {request.customer_id}")

    try:
        if info["backend"] == "ollama":
            # Ollama uses raw prompt with system prompt prepended
            full_prompt = f"{SYSTEM_PROMPT}\n\n{request.prompt}"
            generated_text = await generate_ollama(full_prompt, request)
        else:
            # vLLM uses chat format (system prompt handled in generate_vllm)
            generated_text = await generate_vllm(request.prompt, request, adapter_name=adapter_name)

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


@app.post("/reload-adapter")
async def reload_adapter(request: ReloadAdapterRequest):
    """
    Register a LoRA adapter for a customer.
    In production with vLLM, this would trigger hot-reload of LoRA weights.
    """
    active_adapters[request.customer_id] = request.adapter_path

    info = get_backend_info()

    if info["backend"] == "vllm":
        # In production, we'd call vLLM's LoRA loading endpoint
        # Example: POST /v1/load_lora with adapter_path
        return {
            "status": "ok",
            "message": f"Adapter registered for {request.customer_id}",
            "adapter_path": request.adapter_path,
            "note": "Production: Would hot-reload LoRA into vLLM"
        }
    else:
        return {
            "status": "ok",
            "message": f"Adapter registered for {request.customer_id}",
            "note": "Local mode: Ollama doesn't support runtime LoRA loading"
        }


@app.get("/adapters")
async def list_adapters():
    return {"adapters": active_adapters, "env": ENV}


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
