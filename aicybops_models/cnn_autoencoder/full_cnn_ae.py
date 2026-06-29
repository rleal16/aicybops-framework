import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ContainerMetricsProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.n_features = None
    
    def get_n_features(self):
        return self.n_features
        
    def load_and_process_metrics(self, source):
        """
        Load and process metrics from either a file path or a DataFrame
        Args:
            source: Either a string file path or a pandas DataFrame
        """
        print(f"Loading and processing metrics from {type(source)}")
        
        if isinstance(source, str):
            # Load from file path
            df = pd.read_csv(source, low_memory=False)
        elif isinstance(source, pd.DataFrame):
            # Use the DataFrame directly
            df = source
        else:
            raise ValueError("Source must be either a file path (str) or a pandas DataFrame")
        
        print(f"Loaded dataframe with columns: {df.columns.tolist()}")
        
        numeric_columns = ['counter']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        metrics = df.pivot_table(
            values='counter',
            index='_time',
            columns=['id', 'operation'],
            aggfunc='first'
        ).fillna(0)
        
        metrics.index = pd.to_datetime(metrics.index)
        metrics = metrics.sort_index()
        scaled_data = self.scaler.fit_transform(metrics)
        self.n_features = metrics.shape[1]
        return metrics, scaled_data

class MicroserviceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        print(f"Dataset created with {len(self)} sequences of length {seq_length}")

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_length]

class CNNAutoencoder(nn.Module):
    def __init__(self, n_features, seq_length):
        super(CNNAutoencoder, self).__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_features, kernel_size=3, padding=1)
        )
        
        print(f"Model initialized with {n_features} features and sequence length {seq_length}")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.transpose(1, 2)

class CNNAnomalyDetector:
    def __init__(self, n_features, seq_length):
        self.model = CNNAutoencoder(n_features, seq_length)
        self.threshold = None
        self.reconstruction_errors = None
        
    def train_model(self, dataloader, n_epochs=50):
        print(f"Starting model training for {n_epochs} epochs...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        final_loss = 0.0
        
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            
            avg_loss = train_loss/batch_count
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')
            final_loss = avg_loss
        
        return {"train_loss": float(final_loss)}
    
    def detect_anomalies(self, dataset):
        print("Starting anomaly detection...")
        self.model.eval()
        self.reconstruction_errors = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                seq = dataset[i].unsqueeze(0)
                reconstructed_seq = self.model(seq)
                error = nn.MSELoss(reduction='none')(reconstructed_seq, seq).mean().item()
                self.reconstruction_errors.append(error)
        
        self.threshold = np.mean(self.reconstruction_errors) + 2 * np.std(self.reconstruction_errors)
        detected_anomalies = np.where(np.array(self.reconstruction_errors) > self.threshold)[0]
        
        return detected_anomalies
    
    def visualize_results(self, detected_anomalies):
        plt.figure(figsize=(15, 5))
        plt.plot(self.reconstruction_errors, label='Reconstruction Error')
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        plt.scatter(detected_anomalies, 
                   [self.reconstruction_errors[i] for i in detected_anomalies], 
                   color='red', marker='x', label='Detected Anomalies')
        plt.legend()
        plt.title('Anomaly Detection in Container Metrics')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.show()

if __name__ == "__main__":
    from ..cnn_autoencoder import ContainerAnomalyDetector as CNNAnomalyDetector
    # Initialize processors
    processor = ContainerMetricsProcessor()
    df, scaled_data = processor.load_and_process_metrics('main_test.csv')
    
    # Create dataset and dataloader
    seq_length = 10
    dataset = MicroserviceDataset(scaled_data, seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize detector and train
    detector = CNNAnomalyDetector(n_features=df.shape[1], seq_length=seq_length)
    detector.train_model(dataloader)
    
    # Detect and visualize anomalies
    detected_anomalies = detector.detect_anomalies(dataset)
    detector.visualize_results(detected_anomalies)