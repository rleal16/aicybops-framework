import os
import mlflow
from minio import Minio
from minio.error import S3Error
from urllib.parse import urlparse
import logging
import pytest
import numpy as np

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def mlflow_client():
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5001')
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient()

@pytest.fixture(scope="module")
def minio_client():
    endpoint_url = os.environ.get('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
    access_key = os.environ.get('AWS_ACCESS_KEY_ID', 'minio')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', 'minio123')
    
    parsed_url = urlparse(endpoint_url)
    return Minio(
        parsed_url.netloc,
        access_key=access_key,
        secret_key=secret_key,
        secure=parsed_url.scheme == 'https'
    )

def test_mlflow_connection(mlflow_client):
    logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    try:
        experiment = mlflow_client.get_experiment_by_name("Default")
        assert experiment is not None, "Default experiment not found"
        logger.info(f"Successfully connected to MLflow server. Default experiment ID: {experiment.experiment_id}")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow server: {e}")
        raise

def test_minio_connection(minio_client):
    try:
        buckets = minio_client.list_buckets()
        logger.info(f"Successfully connected to MinIO. Buckets: {[bucket.name for bucket in buckets]}")
        assert len(buckets) > 0, "No buckets found in MinIO"
    except S3Error as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        raise

def test_mlflow_minio_integration(mlflow_client, minio_client):
    experiment_name = "minio_integration_test"
    try:
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow_client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow_client.log_param(run.info.run_id, "test_param", "test_value")
            mlflow_client.log_metric(run.info.run_id, "test_metric", 1.0)
            with open("test_artifact.txt", "w") as f:
                f.write("This is a test artifact")
            mlflow_client.log_artifact(run.info.run_id, "test_artifact.txt")

        artifact_uri = mlflow_client.get_run(run.info.run_id).info.artifact_uri
        parsed_uri = urlparse(artifact_uri)
        bucket_name = parsed_uri.netloc
        artifact_path = parsed_uri.path.lstrip('/')
        try:
            minio_client.stat_object(bucket_name, f"{artifact_path}/test_artifact.txt")
        except S3Error as e:
            logger.error(f"Failed to find artifact in MinIO: {e}")
            raise
    except Exception as e:
        logger.error(f"Error during MLflow-MinIO integration test: {e}")
        raise
    finally:
        if os.path.exists("test_artifact.txt"):
            os.remove("test_artifact.txt")

def test_mlflow_minio_model_storage(mlflow_client, minio_client):
    experiment_name = "model_storage_test"
    try:
        import torch
        import torch.nn as nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
            def forward(self, x):
                return self.linear(x)

        model = TinyModel()
        example_input = np.array([[1.0]])

        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow_client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id) as run:
            model_path = "test_model.pth"
            torch.save(model.state_dict(), model_path)
            mlflow.pytorch.log_model(model, "model", input_example=example_input)
            mlflow_client.log_artifact(run.info.run_id, model_path)

        artifact_uri = mlflow_client.get_run(run.info.run_id).info.artifact_uri
        parsed_uri = urlparse(artifact_uri)
        bucket_name = parsed_uri.netloc
        artifact_path = parsed_uri.path.lstrip('/')
        try:
            objects = list(minio_client.list_objects(bucket_name, prefix=artifact_path, recursive=True))
            assert any('model' in obj.object_name for obj in objects), "Model not found in MinIO"
        except S3Error as e:
            logger.error(f"Failed to find model in MinIO: {e}")
            raise
    except Exception as e:
        logger.error(f"Error during model storage test: {e}")
        raise
    finally:
        if os.path.exists("test_model.pth"):
            os.remove("test_model.pth")
