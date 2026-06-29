# syntax=docker/dockerfile:1.7
# PyTorch variant: "cpu" (default, no NVIDIA driver required) or "cuda" (GPU hosts).
# CPU:  docker compose build
# GPU:  docker compose build --build-arg TORCH_VARIANT=cuda
#       (host needs NVIDIA driver + nvidia-container-toolkit; set DAM_DEVICE=cuda)
ARG TORCH_VARIANT=cpu

FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y git \
    build-essential \
    python3-psycopg2 \
    libpq-dev \
    postgresql-client \
    curl \
    && pip install psycopg2-binary boto3 python-dotenv \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency inputs first for better layer caching
COPY aicybops-lib /aicybops_lib
COPY aicybops-service/requirements.txt .

# Install deps with BuildKit pip cache
RUN --mount=type=cache,target=/root/.cache/pip pip install /aicybops_lib

# Install service requirements; default image is CPU-only (see TORCH_VARIANT above).
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
ARG TORCH_VARIANT
RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
      pip uninstall -y torch triton 2>/dev/null || true; \
      pip freeze | sed -n 's/^\(nvidia-[^=]*\).*/\1/p' | xargs -r pip uninstall -y 2>/dev/null || true; \
      pip uninstall -y cuda-bindings cuda-toolkit cuda-pathfinder 2>/dev/null || true; \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Copy frequently changing application code after dependencies
COPY aicybops_models /app/aicybops_models
COPY aicybops-service/data /app/data
COPY aicybops-service/src/main.py .
COPY aicybops-service/src/worker.py .

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR="/app/data"
ENV AICYBOPS_TORCH_VARIANT=${TORCH_VARIANT}
ENV DAM_DEVICE=cpu
ENV TORCHDYNAMO_DISABLE=1
ENV TORCH_COMPILE_DISABLE=1
ENV PYTORCH_DISABLE_AOT_AUTOGRAD=1
ENV CUDA_VISIBLE_DEVICES=

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
