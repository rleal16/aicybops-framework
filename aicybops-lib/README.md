# aicybops-lib

`aicybops-lib` is the shared library used by AICybOps model integrations and services.
It provides:

- a base model abstraction and model registry
- MLflow tracking/logging helpers
- a FastAPI server layer for train/predict/evaluate flows
- a Python client for interacting with that server

## Current repository layout

```text
aicybops-lib/
├── pyproject.toml
├── README.md
├── src/aicybops_lib/
│   ├── __init__.py
│   ├── base_model/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── registry.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── mlflow_logging.py
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── job_store_redis.py
│   │   ├── monitoring.py
│   │   ├── serialization.py
│   │   ├── worker.py
│   │   └── utils/
│   │       └── __init__.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── config.py
│   └── utils/
│       └── __init__.py
└── tests/
```

## Install

From the `aicybops-lib` directory:

```bash
pip install -e .
```

For test dependencies:

```bash
pip install -e ".[test]"
```

