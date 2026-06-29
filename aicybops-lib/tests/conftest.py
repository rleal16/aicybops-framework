import pytest
import os

@pytest.fixture(scope='session', autouse=True)
def set_env_variables():
    os.environ.update({
        'AWS_ACCESS_KEY_ID': 'minio',
        'AWS_SECRET_ACCESS_KEY': 'minio123',
        'MLFLOW_TRACKING_URI': 'http://localhost:5001',
        'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000',
        'MLFLOW_S3_IGNORE_TLS': 'true',
        'MLFLOW_TRACKING_INSECURE_TLS': 'true',
        'DATA_SOURCE': 'api',
        'API_URL': 'http://213.30.51.238:5010'
    })
