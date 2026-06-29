import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from aicybops_lib.base_model import BaseModel, with_mlflow_logging
from .full_cnn_ae import ContainerMetricsProcessor, MicroserviceDataset, CNNAutoencoder
import numpy as np
from torch.utils.data import DataLoader
from .call_api import CallAPI

class ContainerAnomalyDetector(BaseModel):
    def __init__(self, experiment_name, n_features=None, seq_length=10, **kwargs):
        tracking_uri = kwargs.pop("tracking_uri", os.getenv('MLFLOW_TRACKING_URI'))
        super().__init__(experiment_name, tracking_uri, **kwargs)
        self.n_features = n_features
        self.seq_length = seq_length
        self.model = None
        self.processor = ContainerMetricsProcessor()
        self.dataset = None
        self.trainloader = None
        self.testloader = None
        self.threshold = None
        self.reconstruction_errors = None
        self.validationloader = None
        
        # Get data directory from environment variable
        self.data_dir = Path(os.getenv('LOCAL_DATA_SOURCE', 'aicybops_models/tests/data/'))
        if not self.data_dir.exists():
            raise RuntimeError(f"Data directory not found at {self.data_dir}")
        
        # Default to main_test.csv in the data directory
        self.data_path = self.data_dir / os.getenv('CNN_AUTOENCODER_DATA_SOURCE', 'main_test.csv')
        if not self.data_path.exists():
            raise RuntimeError(f"CNN_AUTOENCODER_DATA_SOURCE file not found at {self.data_path}")

    def _resolve_data_path(self):
        """Resolve the data path from LOCAL_DATA_SOURCE environment variable"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        return str(self.data_path)

    def get_training_data(self, **kwargs):
        """
        Get training data loader.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary (accepted for API consistency, but uses internal state).
        
        Returns:
            Training data loader
        """
        # Accept data for API consistency, but use internal state
        if self.trainloader is None:
            self._prepare_data()
        return self.trainloader

    def get_test_data(self, **kwargs):
        """
        Get test data loader.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary (accepted for API consistency, but uses internal state).
        
        Returns:
            Test data loader
        """
        # Accept data for API consistency, but use internal state
        if self.testloader is None:
            self._prepare_data()
        return self.testloader
    
    def get_validation_data(self, **kwargs):
        """
        Get validation data loader.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary (accepted for API consistency, but uses internal state).
        
        Returns:
            Validation data loader
        """
        # Accept data for API consistency, but use internal state
        if self.validationloader is None:
            self._prepare_data()
        return self.validationloader
    
    def get_prediction_data(self, **kwargs):
        """
        Get prediction data.
        
        Args:
            **kwargs: Optional parameters including:
                - data: Data dictionary (accepted for API consistency, but uses internal state).
        
        Returns:
            Prediction data
        """
        # Accept data for API consistency, but use internal state
        return self.fetch_data_for_prediction()

    def _split_and_create_loaders(self, source_df):

        df, scaled_data = self.processor.load_and_process_metrics(source_df)
                    
        self.n_features = df.shape[1]
        self.dataset = MicroserviceDataset(scaled_data, self.seq_length)

        # Calculate sizes ensuring they sum to total length
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        test_size = int(0.15 * total_size)
        val_size = total_size - (train_size + test_size)

        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, 
            [train_size, test_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        self.validationloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return df, scaled_data, self.dataset

    # only uses API
    def _get_data_from_api(self, start: str = '-120', save_to_disk: bool = False):
        try:
            
            api_url = os.getenv('API_URL')
            if not api_url:
                raise ValueError("API_URL environment variable must be set.")
            
            call_api = CallAPI(api_url)
            df = call_api.all_container_metrics(start=start, save_to_disk=save_to_disk)
            if df is None:
                raise ValueError("Failed to fetch data from API")
                
            return self._split_and_create_loaders(df)
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

    # can use local or api
    def _prepare_data(self, fallback_to_local_data: bool = False):
        try:
            data_source = os.getenv('MAIN_DATA_SOURCE', 'local')
            
            if data_source.lower() == 'api':
                return self._get_data_from_api()
            else:
                resolved_path = self._resolve_data_path()
                return self._split_and_create_loaders(resolved_path)
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            if fallback_to_local_data and data_source.lower() == 'api':
                print(f"Falling back to local data source")
                try:
                    resolved_path = self._resolve_data_path()
                    return self._split_and_create_loaders(resolved_path)
                except Exception as fallback_error:
                    print(f"Fallback to local data source failed: {fallback_error}")
            
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def build_model(self):
        if self.n_features is None:
            self._prepare_data()
        self.model = CNNAutoencoder(self.n_features, self.seq_length)
    
    @with_mlflow_logging
    def train(self, **kwargs) -> dict:
        # Extract params and epochs from kwargs
        params = kwargs.get('params', {})
        epochs = kwargs.get('epochs')
        if epochs is None:
            raise ValueError("epochs parameter required. Pass epochs=<number> in kwargs.")
        if self.model is None:
            self.build_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=params.get('lr', 0.001))
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in self.trainloader:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_loss = train_loss/len(self.trainloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        return {"train_loss": avg_loss}
    
    
    def _evaluate(self, data_loader) -> dict:
        self.model.eval()
        self.reconstruction_errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                reconstructed_seq = self.model(batch)
                error = nn.MSELoss(reduction='none')(reconstructed_seq, batch).mean(dim=1)
                self.reconstruction_errors.extend(error.tolist())
        
        mean_reconstruction_error = np.mean(self.reconstruction_errors)
        std_reconstruction_error = np.std(self.reconstruction_errors)
        self.threshold = mean_reconstruction_error + 2 * std_reconstruction_error
        detected_anomalies = np.where(np.array(self.reconstruction_errors) > self.threshold)[0]
        
        return {
            "anomalies_detected": len(detected_anomalies),
            "threshold": self.threshold,
            "mean_reconstruction_error": mean_reconstruction_error
        }
    
    @with_mlflow_logging
    def test(self, **kwargs) -> dict:
        test_loader = self.get_test_data()
        return self._evaluate(test_loader)
    
    @with_mlflow_logging
    def validate(self, **kwargs) -> dict:
        val_loader = self.get_validation_data()
        return self._evaluate(val_loader)
    
    def get_model_metrics(self) -> dict:
        return {
            'prediction': {'metric': 'mean_reconstruction_error', 'mode': 'min'}, 
            'training': ['train_loss'],
            'evaluation': ['anomalies_detected', 'threshold', 'mean_reconstruction_error']
        }
        
    
    def predict(self, model, model_info = None, data = None):
        # Use data parameter if provided, otherwise get from get_prediction_data
        if data is None:
            data = self.get_prediction_data()
        else:
            # If data is provided, pass it to get_prediction_data for consistency
            data = self.get_prediction_data(data=data)
        
        if data is None:
            raise ValueError("No data path provided for prediction")
        
        threshold = None
        if model_info is not None and 'metrics.threshold' in model_info:
            threshold = model_info['metrics.threshold']
        elif hasattr(self, 'threshold'):
            threshold = self.threshold
        
        if threshold is None:
            raise ValueError("Threshold not set. Model needs to be trained first")
        
        print(f"Model info: {model_info}")

        model.eval()
        with torch.no_grad():
            seq = torch.FloatTensor(data).unsqueeze(0)
            reconstructed_seq = model(seq)
            error = nn.MSELoss(reduction='none')(reconstructed_seq, seq).mean().item()
            return error > threshold



    #def get_example_input(self):
    #    return torch.randn(1, self.seq_length, self.n_features)
    
    def get_example_input(self):
        if self.n_features is None:
            # Initialize model first to set n_features
            self.build_model()
            
        return torch.randn(1, self.seq_length, self.n_features)

    def fetch_data_for_prediction(self):
                    
        # Use the same processor and data preparation as in training
        _, scaled_data, _ = self._prepare_data()
        # Create a sequence of the last seq_length points
        prediction_data = scaled_data[-self.seq_length:]
        return prediction_data