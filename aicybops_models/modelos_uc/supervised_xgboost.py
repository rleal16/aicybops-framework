import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GridSearchCV, GroupKFold
import xgboost as xgb
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
BASE_DIR = Path(__file__).resolve().parent


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

        training = pd.read_csv(BASE_DIR / "training_data.csv")
        training.loc[training["momentum"] == "pre", "type"] = "normal"
        training.loc[training["momentum"] == "pos", "type"] = "normal"
        training["type"] = np.where(training["type"] == "normal", 0, 1)
        y_train = training["type"]
        X_train = training.drop(columns=columns_to_drop, axis=1)

        sc = StandardScaler()
        X_train = pd.DataFrame(sc.fit_transform(X_train),columns = X_train.columns)

        # model
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        model = xgb.XGBClassifier(random_state=42)

        param_grid = {
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.1, 0.01, 0.001],
                        'subsample': [0.5, 0.7, 1]
                        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_


        # SECOND: testing with attack runs
        testing = pd.read_csv(BASE_DIR / "testing_data.csv")
        testing.loc[testing["momentum"] == "pre", "type"] = "normal"
        testing.loc[testing["momentum"] == "pos", "type"] = "normal"
        testing["type"] = np.where(testing["type"] == "normal", 0, 1)
        y_test = testing["type"]
        X_test = testing.drop(columns=columns_to_drop, axis=1)
        X_test = pd.DataFrame(sc.transform(X_test),columns = X_test.columns)

        # model
        y_pred = best_model.predict(X_test)

        # performance metrics
        print(classification_report(y_test, y_pred, target_names=['normal', 'attack'], zero_division=0))

        return {
                "model": model,
                "best_model": best_model,
                "scaler": sc,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred": y_pred,
                "X_train": X_train,
                "X_test": X_test,
                "grid_search": grid_search,
        }


if __name__ == "__main__":
        run_pipeline()
