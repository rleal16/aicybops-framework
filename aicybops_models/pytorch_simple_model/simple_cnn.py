import torch
import torch.nn as nn
import torch.optim as optim
from aicybops_lib.base_model.base_model import BaseModel, with_mlflow_logging
import numpy as np


class SimpleCNN(BaseModel):
    def __init__(self, experiment_name, train_data=None, test_data=None, data_path=None, **kwargs):
        super().__init__(experiment_name, tracking_uri=kwargs.pop("tracking_uri", data_path), **kwargs)
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.train_data = train_data
        self.test_data = test_data
        self.trainloader = None
        self.testloader = None
        self.validationloader = None

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
    
    def _prepare_data(self, train_data=None, test_data=None):
        # Use instance variables if no parameters provided
        if train_data is None:
            train_data = self.train_data
        if test_data is None:
            test_data = self.test_data

        # If no data available, generate synthetic data
        if train_data is None and test_data is None:
            total_samples = 140  # 100 train + 20 test + 20 validation
            all_data = (
                torch.randn(total_samples, 3, 32, 32),
                torch.randint(0, 10, (total_samples,))
            )
            
            # Split sizes
            train_size = int(0.7 * total_samples)
            test_size = int(0.2 * total_samples)
            validation_size = total_samples - train_size - test_size
            
            # Split data
            inputs, labels = all_data
            train_data = (inputs[:train_size], labels[:train_size])
            test_data = (inputs[train_size:train_size+test_size], labels[train_size:train_size+test_size])
            validation_data = (inputs[train_size+test_size:], labels[train_size+test_size:])
        else:
            # Use provided data and create validation split from train data
            train_inputs, train_labels = train_data
            validation_size = int(0.15 * len(train_inputs))
            train_size = len(train_inputs) - validation_size
            
            train_data = (train_inputs[:train_size], train_labels[:train_size])
            validation_data = (train_inputs[train_size:], train_labels[train_size:])

        # Create datasets
        train_inputs, train_labels = train_data
        test_inputs, test_labels = test_data
        validation_inputs, validation_labels = validation_data

        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_inputs, test_labels)
        validation_dataset = torch.utils.data.TensorDataset(validation_inputs, validation_labels)

        # Create dataloaders
        self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
        self.validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=False)

    def build_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        # Determine the size of the flattened feature vector
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 32, 32)
            flattened_size = self.model(sample_input).shape[1]
        
        # Complete the model with the determined flattened size
        self.model = nn.Sequential(
            self.model,
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    @with_mlflow_logging
    def train(self, **kwargs) -> dict:
        # Extract params and epochs from kwargs
        params = kwargs.get('params', {})
        epochs = kwargs.get('epochs')
        if epochs is None:
            raise ValueError("epochs parameter required. Pass epochs=<number> in kwargs.")
        if self.model is None:
            self.build_model()
        train_data = self.get_training_data(data=kwargs.get('data'))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        final_loss = 0.0
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_data, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                    final_loss = running_loss / 2000
                    running_loss = 0.0
        return {"train_loss": final_loss}
    

    # helper function for validation and testing
    def _evaluate(self, loader, prefix=""):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return {f"{prefix}_accuracy": accuracy}

    @with_mlflow_logging
    def validate(self, **kwargs) -> dict:
        self.model.eval()
        validation_data = self.get_validation_data(data=kwargs.get('data'))
        metrics = self._evaluate(validation_data, "validation")
        return metrics
        
    @with_mlflow_logging
    def test(self, **kwargs) -> dict:
        test_data = self.get_test_data(data=kwargs.get('data'))
        return self._evaluate(test_data, "test")

    def get_model_metrics(self) -> dict:
        return {
            'prediction': {'metric': 'validation_accuracy', 'mode': 'max'},
            'training': ['train_loss'],
            'validation': ['validation_accuracy'],
            'testing': ['test_accuracy']
        }

    # placeholder where, later, the data will be fetched from the metrics database
    def fetch_data_for_prediction(self, data_path=None):
        print("Fetching data for prediction")
        batch_size = 5
        channels = 3
        height = 32
        width = 32
        predict_data = np.random.rand(batch_size, channels, height, width).tolist()
        print(f"Predict data shape: {np.array(predict_data).shape}")
        return predict_data

    def predict(self, model, model_info = None):
        data = self.get_prediction_data()
        print(f"Model is instance of {type(model)}")
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(data, dtype=torch.float32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
    

    def get_example_input(self):
        if self.train_data is not None:
            train_inputs, _ = self.train_data
            return train_inputs[0:1]
        else:
            return torch.randn(1, 3, 32, 32)
        
