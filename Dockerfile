FROM python:3.11-slim

WORKDIR /app

# Install dependencies (FastAPI service only, vLLM runs natively on VM)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    python-dotenv \
    pydantic

# Copy application code
COPY app/ ./app/

# Set environment
ENV ENV=production
ENV VLLM_URL=http://localhost:8080
ENV VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
