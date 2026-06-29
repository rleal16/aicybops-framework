from typing import List
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import OneClassSVM, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET_FILEPATH = "data/dataset_BoSC.dat"
MODELOS_UC2_ROOT = Path(__file__).resolve().parent

MODELS = {
    "decision_tree": {"name": "Decision Tree", "data": DEFAULT_DATASET_FILEPATH},
    "random_forest": {"name": "Random Forest", "data": DEFAULT_DATASET_FILEPATH},
    "gaussian_nb": {"name": "Gaussian Naive Bayes", "data": DEFAULT_DATASET_FILEPATH},
    "complement_nb": {"name": "Complement Naive Bayes", "data": DEFAULT_DATASET_FILEPATH},
    "svm": {"name": "SVM", "data": DEFAULT_DATASET_FILEPATH},
    "one_class_svm": {"name": "One-Class SVM", "data": DEFAULT_DATASET_FILEPATH},
    "mlp": {"name": "MLP", "data": DEFAULT_DATASET_FILEPATH},
}

def _format_output(
    model_name,
    mode,
    output_format,
    y_test=None,
    y_pred=None,
    precision_anomaly=None,
    recall_anomaly=None,
    f1_anomaly=None,
    confusion_matrix_avg=None,
):
    if output_format == "legacy":
        return (y_test, y_pred) if mode == "supervised" else None
    if output_format != "normalized":
        raise ValueError("output_format must be either 'legacy' or 'normalized'")

    if mode == "supervised":
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
    elif mode == "one_class":
        metrics = {
            "precision_anomaly": float(precision_anomaly) if precision_anomaly is not None else None,
            "recall_anomaly": float(recall_anomaly) if recall_anomaly is not None else None,
            "f1_anomaly": float(f1_anomaly) if f1_anomaly is not None else None,
            "confusion_matrix_avg": np.asarray(confusion_matrix_avg).tolist() if confusion_matrix_avg is not None else None,
        }
    else:
        raise ValueError("mode must be either 'supervised' or 'one_class'")

    return {
        "mode": mode,
        "model_name": model_name,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }


# Selects a model from a list of preconfigured models and uses the BoSC dataset to train and evaluate it, returning y_test and y_pred.  
def train_test_model(model_name, dataset_filepath, output_format="legacy"):
    dataset = pd.read_csv(dataset_filepath, sep=r"\s+")

    X = dataset.drop(columns=["label"]).to_numpy()
    y = dataset["label"].to_numpy()

    # Creates an object using Stratified K-Fold, enabling k-fold cross-validation with k = 10
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)                   # Used for classification models
    kf = KFold(n_splits=10, shuffle=True, random_state=32)                              # Used for anomaly detection models

    rng = np.random.default_rng(32)

    all_y_test = []
    all_y_pred = []

    # Select which classifier to use to build, train, and evaluate the model. They are calculated for each fold, but the results are aggregated across all folds.
    match model_name:
        case "Decision Tree":
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = DecisionTreeClassifier(
                    random_state=32,
                    class_weight="balanced"  # optional but helpful for imbalance
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                all_y_test.append(y_test)
                all_y_pred.append(y_pred)

            all_y_test = np.concatenate(all_y_test)
            all_y_pred = np.concatenate(all_y_pred)

            return _format_output(
                model_name=model_name,
                mode="supervised",
                output_format=output_format,
                y_test=all_y_test,
                y_pred=all_y_pred,
            )
        case "Random Forest":
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            
                model = RandomForestClassifier(random_state=32)
                model.fit(X_train, y_train)
            
                y_pred = model.predict(X_test)
            
                all_y_test.append(y_test)
                all_y_pred.append(y_pred)
            
            all_y_test = np.concatenate(all_y_test)
            all_y_pred = np.concatenate(all_y_pred)

            return _format_output(
                model_name=model_name,
                mode="supervised",
                output_format=output_format,
                y_test=all_y_test,
                y_pred=all_y_pred,
            )
        case "Gaussian Naive Bayes":
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = GaussianNB()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                all_y_test.append(y_test)
                all_y_pred.append(y_pred)

            all_y_test = np.concatenate(all_y_test)
            all_y_pred = np.concatenate(all_y_pred)

            return _format_output(
                model_name=model_name,
                mode="supervised",
                output_format=output_format,
                y_test=all_y_test,
                y_pred=all_y_pred,
            )
        case "Complement Naive Bayes":
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = ComplementNB(alpha=1.0)  # alpha = smoothing
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                all_y_test.append(y_test)
                all_y_pred.append(y_pred)

            all_y_test = np.concatenate(all_y_test)
            all_y_pred = np.concatenate(all_y_pred)

            return _format_output(
                model_name=model_name,
                mode="supervised",
                output_format=output_format,
                y_test=all_y_test,
                y_pred=all_y_pred,
            )
        case "SVM":
            sample_frac = 0.3  # 30% of the dataset
            sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_frac, random_state=32)

            subset_idx, _ = next(sss.split(X, y))
            X_SVM = X[subset_idx]
            y_SVM = y[subset_idx]

            for train_idx, test_idx in skf.split(X_SVM, y_SVM):
                X_train, X_test = X_SVM[train_idx], X_SVM[test_idx]
                y_train, y_test = y_SVM[train_idx], y_SVM[test_idx]

                model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", gamma="scale", C=1.0, random_state=32))])

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                all_y_test.append(y_test)
                all_y_pred.append(y_pred)

            all_y_test = np.concatenate(all_y_test)
            all_y_pred = np.concatenate(all_y_pred)

            return _format_output(
                model_name=model_name,
                mode="supervised",
                output_format=output_format,
                y_test=all_y_test,
                y_pred=all_y_pred,
            )
        case "One-Class SVM":
            normal_label = 0
            precisions = []
            recalls = []
            f1s = []
            conf_matrices = []
        
            X_normal = X[y == normal_label]
            X_anomaly = X[y != normal_label]

            max_normals = 20000   # try 5000–20000
            if len(X_normal) > max_normals:
                idx = rng.choice(len(X_normal), size=max_normals, replace=False)
                X_normal_small = X_normal[idx]
            else:
                X_normal_small = X_normal

            for train_idx, test_idx in kf.split(X_normal_small):
                # Train only with normal data
                X_train = X_normal_small[train_idx]

                # Test with normal and anomaly data
                X_test_normal = X_normal_small[test_idx]

                # Evaluate on held-out normals + (optionally sampled) anomalies
                max_anoms = 5000
                if len(X_anomaly) > max_anoms:
                    aidx = rng.choice(len(X_anomaly), size=max_anoms, replace=False)
                    X_anom_eval = X_anomaly[aidx]
                else:
                    X_anom_eval = X_anomaly

                X_test = np.vstack([X_test_normal, X_anom_eval])

                # Ground-truth labels for evaluation: 1 = normal, -1 = anomaly
                y_test = np.hstack([np.ones(len(X_test_normal), dtype=int), -np.ones(len(X_anom_eval), dtype=int)])
                
                model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
                model.fit(X_train)

                # OneClassSVM predicts: 1 (inlier/normal), -1 (outlier/anomaly)
                y_pred = model.predict(X_test)

                # --- per-fold metrics ---
                precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
                recall = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
                f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)

                cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                conf_matrices.append(cm)

            mean_precision = np.mean(precisions)
            mean_recall = np.mean(recalls)
            mean_f1 = np.mean(f1s)
            mean_cm = np.mean(conf_matrices, axis=0)

            print(f"Precision (anomaly): {mean_precision:.3f}")
            print(f"Recall (anomaly):    {mean_recall:.3f}")
            print(f"F1-score (anomaly):  {mean_f1:.3f}")

            print("\nAverage confusion matrix (per fold):")
            print(mean_cm)

            return _format_output(
                model_name=model_name,
                mode="one_class",
                output_format=output_format,
                y_test=None,
                y_pred=None,
                precision_anomaly=mean_precision,
                recall_anomaly=mean_recall,
                f1_anomaly=mean_f1,
                confusion_matrix_avg=mean_cm,
            )
        case "MLP":
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = Pipeline([("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,          # L2 regularization
                        batch_size=256,
                        learning_rate_init=1e-3,
                        max_iter=50,         # keep low for speed
                        random_state=32,
                        early_stopping=True,
                        n_iter_no_change=5,
                        verbose=False
                    ))
                ])

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                all_y_test.append(y_test)
                all_y_pred.append(y_pred)

            all_y_test = np.concatenate(all_y_test)
            all_y_pred = np.concatenate(all_y_pred)

            return _format_output(
                model_name=model_name,
                mode="supervised",
                output_format=output_format,
                y_test=all_y_test,
                y_pred=all_y_pred,
            )

        
def metrics(results_test, results_pred=None):
    if isinstance(results_test, dict):
        out = results_test
        mode = out.get("mode")
        model_name = out.get("model_name", "unknown")
        metric_data = out.get("metrics", {})
        print(f"Model: {model_name}")
        if mode == "supervised":
            print("Global accuracy:", metric_data.get("accuracy"))
            print(classification_report(out.get("y_test"), out.get("y_pred"), zero_division=0))
            print(confusion_matrix(out.get("y_test"), out.get("y_pred")))
        elif mode == "one_class":
            print("Precision (anomaly):", metric_data.get("precision_anomaly"))
            print("Recall (anomaly):", metric_data.get("recall_anomaly"))
            print("F1-score (anomaly):", metric_data.get("f1_anomaly"))
            print("Average confusion matrix (per fold):")
            print(np.asarray(metric_data.get("confusion_matrix_avg")))
        return metric_data

    print("Global accuracy:", accuracy_score(results_test, results_pred))
    print(classification_report(results_test, results_pred))
    print(confusion_matrix(results_test, results_pred))
    return {
        "accuracy": float(accuracy_score(results_test, results_pred)),
        "classification_report": classification_report(results_test, results_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(results_test, results_pred).tolist(),
    }


def run_pipeline(model_names: Optional[List[str]] = None, output_format: str = "normalized"):
    """Run all modelos_uc_2 models (or a selected subset) and return metrics per model.

    model_names accepts keys from MODELS (e.g. "decision_tree", "svm", "one_class_svm").
    By default, all configured models are executed in MODELS order.
    """
    selected_keys = model_names or list(MODELS.keys())
    unknown = [k for k in selected_keys if k not in MODELS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {list(MODELS.keys())}")

    outputs: Dict[str, Dict[str, Any]] = {}
    for model_key in selected_keys:
        cfg = MODELS[model_key]
        model_name = cfg["name"]
        dataset_path = str(MODELOS_UC2_ROOT / cfg["data"])

        result = train_test_model(model_name, dataset_path, output_format=output_format)
        if output_format == "legacy":
            if result is None:
                model_metrics = {}
            else:
                results_test, results_pred = result
                model_metrics = metrics(results_test, results_pred)
        else:
            model_metrics = metrics(result)
        outputs[model_name] = model_metrics

    return outputs



if __name__ == "__main__":
    run_pipeline()