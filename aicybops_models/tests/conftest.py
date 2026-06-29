import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture(scope='session', autouse=True)
def set_env_variables():
    # Set data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create test data file
    test_file = data_dir / "main_test.csv"
    if not test_file.exists():
        df = pd.DataFrame(np.random.randn(100, 10), 
                         columns=[f'metric_{i}' for i in range(10)])
        df.to_csv(test_file, index=False)
    
    # Set environment variables for MLflow - using the correct URL format
    os.environ.update({
        'DATA_DIR': str(data_dir),
        'MLFLOW_TRACKING_URI': 'http://localhost:5001',
        'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000',
        'AWS_ACCESS_KEY_ID': 'minio',
        'AWS_SECRET_ACCESS_KEY': 'minio123',
        'MLFLOW_S3_IGNORE_TLS': 'true',
        'MLFLOW_TRACKING_INSECURE_TLS': 'true',
        'DATA_SOURCE': 'api',
        'API_URL': 'http://213.30.51.238:5010'
    })

@pytest.fixture(params=["main_test.csv"])
def test_data_path(request):
    data_dir = Path(os.environ['DATA_DIR'])
    data_file = data_dir / request.param
    if not data_file.exists():
        pytest.skip(f"Test data file {data_file} not found")
    return data_file 