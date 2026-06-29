FROM ghcr.io/mlflow/mlflow:latest

RUN apt-get update && apt-get install -y \
    python3-psycopg2 \
    libpq-dev \
    postgresql-client \
    && pip install psycopg2-binary boto3 python-dotenv \
    && rm -rf /var/lib/apt/lists/*