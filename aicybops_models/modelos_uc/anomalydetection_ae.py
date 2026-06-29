import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.config.experimental.enable_op_determinism()
BASE_DIR = Path(__file__).resolve().parent

def build_autoencoder(input_dim):
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        
        # Bottleneck
        encoded = Dense(4, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        # decoded = Dense(input_dim, activation='sigmoid')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse') 
        
        
        return autoencoder

columns_to_drop = ['type', 'time', 'timestamp', 'momentum', 'cpu_usage_webui', 'memory_working_webui',
        'memory_max_webui', 'memory_rss_webui',
        'cpu_usage_auth', 'memory_working_auth',
        'memory_max_auth', 'memory_rss_auth',
        'cpu_usage_image', 'memory_working_image',
        'memory_max_image', 'memory_rss_image',
        'cpu_usage_persistence', 'memory_working_persistence',
        'memory_max_persistence', 'memory_rss_persistence',
        'cpu_usage_recommender','memory_working_recommender',
        'memory_max_recommender', 'memory_rss_recommender']


def run_pipeline():
        # FIRST: tranining with golden runs

        training = pd.read_csv(BASE_DIR / "golden_runs.csv")
        X_train = training.drop(columns=columns_to_drop, axis=1)

        sc = StandardScaler()
        X_train = pd.DataFrame(sc.fit_transform(X_train),columns = X_train.columns)

        input_dim = X_train.shape[1]
        model = build_autoencoder(input_dim)

        history = model.fit(
                        X_train, X_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_train, X_train),
                        validation_split=0.2,
                        verbose=0
                    )

        reconstructions = model.predict(X_train)
        train_loss = tf.keras.losses.mae(reconstructions, X_train)
        threshold = np.percentile(train_loss.numpy(), 99)

        # SECOND: testing with attack runs
        testing = pd.read_csv(BASE_DIR / "attack_runs.csv")
        testing.loc[testing["momentum"] == "pre", "type"] = "normal"
        testing.loc[testing["momentum"] == "pos", "type"] = "normal"
        testing["type"] = np.where(testing["type"] == "normal", 0, 1)
        y_test = testing["type"]
        X_test = testing.drop(columns=columns_to_drop, axis=1)
        X_test = pd.DataFrame(sc.transform(X_test),columns = X_test.columns)

        # model
        reconstructions = model.predict(X_test)
        test_loss = tf.keras.losses.mae(reconstructions, X_test).numpy()
        test_loss = tf.keras.losses.mae(reconstructions, X_test).numpy()
        y_pred = (test_loss > threshold).astype(int)  

        # performance metrics
        print("Anomalies found:", np.sum(y_pred))
        print(classification_report(y_test, y_pred, target_names=['normal', 'attack'], zero_division=0))

        return {
                "model": model,
                "scaler": sc,
                "threshold": threshold,
                "history": history,
                "y_test": y_test,
                "y_pred": y_pred,
                "X_train": X_train,
                "X_test": X_test,
        }


if __name__ == "__main__":
        run_pipeline()
