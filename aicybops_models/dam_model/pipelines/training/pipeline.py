import sys
from pathlib import Path

# Add dam_model root and processing paths for imports.
_base_dir = Path(__file__).resolve().parent.parent.parent
if str(_base_dir) not in sys.path:
    sys.path.insert(0, str(_base_dir))
if str(_base_dir / "processing") not in sys.path:
    sys.path.insert(0, str(_base_dir / "processing"))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.detection import SPOT, DriftSPOT
from utils.training import EarlyStopping
from utils.config import RISK_LEVEL, DEPTH, SPOT_TYPE, INIT_QUANTILE
from typing import Tuple, Optional

class DAMPipeline:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, num_epochs=10, device=None,
                 spot_type: str = SPOT_TYPE,        # Choose between "SPOT" or "dSPOT"
                 risk_level: float = RISK_LEVEL,        # The 'q' parameter for EVT
                 depth: int = DEPTH,                 # Window size for moving average in dSPOT
                 init_quantile: float = INIT_QUANTILE,  # Percentile used to seed the EVT threshold
                 early_stopping: Optional[EarlyStopping] = None):  # Early stopping callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.spot_type = spot_type  # Store the detector type
        self.init_quantile = init_quantile  # Store for use in fit_baseline

        # Initialize appropriate detector
        if spot_type == "SPOT":
            self.anomaly_detector = SPOT(risk_level=risk_level)
        elif spot_type == "dSPOT":
            self.anomaly_detector = DriftSPOT(risk_level=risk_level, depth=depth)
        else:
            raise ValueError(f"Unknown spot_type '{spot_type}'. Choose 'SPOT' or 'dSPOT'.")

        self.train_anomaly_scores = None
        self.train_losses = []  # Store per-epoch training losses
        self.final_train_loss = None  # Final training loss from last epoch
        self.early_stopping = early_stopping
        self.best_model_state = None  # Store best model state for early stopping

    @classmethod
    def for_prediction(
        cls,
        model,
        train_loader,
        val_loader,
        spot_type: str = SPOT_TYPE,
        risk_level: float = RISK_LEVEL,
        depth: int = DEPTH,
        init_quantile: float = INIT_QUANTILE,
        device: Optional[torch.device] = None,
    ) -> "DAMPipeline":
        """
        Create a DAMPipeline ready for prediction only (no training).
        Fits the anomaly detector baseline on train_loader so the pipeline can run predict().
        Use when the model was already trained (e.g. loaded from registry) and you only need inference.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        pipeline = cls(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            num_epochs=0,
            device=device,
            spot_type=spot_type,
            risk_level=risk_level,
            depth=depth,
            init_quantile=init_quantile,
        )
        pipeline.train()  # 0 epochs: skips training loop, computes train_anomaly_scores and fits baseline
        return pipeline

    def _compute_loss(self, pred_load, target_load, pred_traffic, target_traffic, pred_log, target_log):
        return (
            self.criterion(pred_load, target_load) +
            self.criterion(pred_traffic, target_traffic) +
            self.criterion(pred_log, target_log)
        )

    def _best_model_so_far(self, epoch):
        if self.early_stopping is None:
            return None
        return self.early_stopping.best_epoch == epoch + 1

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                load_seq, traffic_seq, log_seq, target_load, target_traffic, target_log = [x.to(self.device) for x in batch]
                pred_load, pred_traffic, pred_log = self.model(load_seq, traffic_seq, log_seq)
                loss = self._compute_loss(pred_load, target_load, pred_traffic, target_traffic, pred_log, target_log)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if (i + 1) % 5 == 0 or (i + 1) == len(self.train_loader):
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {i+1}/{len(self.train_loader)}, Batch Loss: {loss.item():.4f}")
                    if epoch == 0 and i == 0:
                        print("Sample prediction (load):", pred_load[0].detach().cpu().numpy())
                        print("Sample target (load):    ", target_load[0].detach().cpu().numpy())

            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)  # Store per-epoch training loss
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Train Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Early stopping check
            if self.early_stopping is not None:
                # Check if should stop (this also updates best_score if improved)
                should_stop = self.early_stopping(val_loss, epoch + 1)

                # Save best model state if this is the best so far
                if self._best_model_so_far(epoch+1):
                    self.best_model_state = {
                        'model_state_dict': self.model.state_dict().copy(),
                        'epoch': epoch + 1
                    }

                # Check if should stop
                if should_stop:
                    print(f"\n[Early Stopping] Stopping training at epoch {epoch+1}")
                    print(f"  Best validation loss: {self.early_stopping.best_score:.4f} at epoch {self.early_stopping.best_epoch}")
                    # Restore best model weights
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state['model_state_dict'])
                        print(f"  Restored model weights from epoch {self.best_model_state['epoch']}")
                    break

        # Store final training loss from last epoch
        self.final_train_loss = self.train_losses[-1] if self.train_losses else None

        # After training, compute anomaly scores on training data for baseline
        print("\n[INFO] Computing anomaly scores on training data for anomaly detector baseline...")
        _, self.train_anomaly_scores = self._predict_and_calculate_anomaly_scores(self.train_loader)

        # Fit DriftSPOT baseline using the configured init_quantile
        self.anomaly_detector.fit_baseline(
            baseline_scores=self.train_anomaly_scores,
            stream_scores=np.array([]),
            init_quantile=self.init_quantile,
        )
        print(f"[INFO] Anomaly detector baseline fitted using {self.spot_type}.")

    def validate(self, data_loader: Optional[DataLoader] = None):
        """
        Validate the model.

        Args:
            data_loader: Optional DataLoader to use for validation. If None, uses self.val_loader.

        Returns:
            Average validation loss
        """
        # Use provided data_loader or fall back to self.val_loader
        if data_loader is None:
            if self.val_loader is None:
                raise ValueError("No validation data loader available. Provide data_loader parameter or set self.val_loader.")
            data_loader = self.val_loader

        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                load_seq, traffic_seq, log_seq, target_load, target_traffic, target_log = [x.to(self.device) for x in batch]
                pred_load, pred_traffic, pred_log = self.model(load_seq, traffic_seq, log_seq)
                loss = self._compute_loss(pred_load, target_load, pred_traffic, target_traffic, pred_log, target_log)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(data_loader)
        return avg_val_loss

    def get_final_train_loss(self):
        """Get the final training loss from the last epoch."""
        return self.final_train_loss

    def get_train_losses(self):
        """Get list of training losses per epoch."""
        return self.train_losses

    def get_train_anomaly_scores(self) -> Optional[np.ndarray]:
        """Get training anomaly scores if available."""
        return self.train_anomaly_scores if self.train_anomaly_scores is not None else None

    def get_baseline_fitted(self) -> bool:
        """Get whether the anomaly detector baseline has been fitted."""
        if self.anomaly_detector is None:
            return False
        return self.anomaly_detector.get_baseline_fitted()

    def get_risk_level(self) -> Optional[float]:
        """Returns the risk_level (q parameter) of the anomaly detector."""
        if self.anomaly_detector is None:
            return None
        return self.anomaly_detector.get_risk_level()

    def get_depth(self) -> Optional[int]:
        """Returns the depth parameter of the anomaly detector (for dSPOT)."""
        if self.anomaly_detector is None:
            return None
        return self.anomaly_detector.get_depth()

    def predict(
        self,
        data_loader=None,
        return_anomaly_scores: bool = False,
        plot_anomaly_scores: bool = False,
        threshold_std: float = 2.0,
        stream_mode: bool = False
    ):
        """
        Run inference on a data loader (or self.val_loader if None).
        Optionally returns anomaly scores, classifications, and/or plots them.

        Args:
            data_loader: DataLoader to use for prediction (defaults to self.val_loader)
            return_anomaly_scores: If True, returns anomaly scores and classifications
            plot_anomaly_scores: If True, plots anomaly scores (batch mode only)
            threshold_std: Standard deviation multiplier for plotting threshold (batch mode only)
            stream_mode: If True, uses streaming anomaly detection; if False, uses batch mode with static threshold

        Returns:
            - If return_anomaly_scores=False: predictions (list)
            - If return_anomaly_scores=True and stream_mode=False:
              (predictions, anomaly_scores, classifications) - 3-tuple
            - If return_anomaly_scores=True and stream_mode=True:
              (predictions, anomaly_scores, classifications, thresholds, alarms) - 5-tuple

        Note:
            Requires model to be trained first (training fits the EVT baseline).
        """
        # Validate EVT detector is fitted when return_anomaly_scores=True
        if return_anomaly_scores and (self.anomaly_detector is None or self.train_anomaly_scores is None):
            raise ValueError("EVT detector not fitted. Model must be trained first (training fits the EVT baseline).")
        if return_anomaly_scores and not self.anomaly_detector.get_baseline_fitted():
            raise ValueError("EVT detector baseline not fitted. Model must be trained first.")

        if data_loader is None:
            data_loader = self.val_loader

        if stream_mode:
            # Streaming mode: use run_anomaly_detection_on_stream()
            stream_results = self.run_anomaly_detection_on_stream(
                stream_loader=data_loader,
                plot_results=plot_anomaly_scores,
                update_with_alarm=False
            )

            # Extract results
            anomaly_scores = stream_results["anomaly_scores"]
            alarms = stream_results["alarms"]
            thresholds = stream_results["thresholds"]
            all_preds = stream_results["predictions"]

            # Convert alarms to binary classifications
            classifications = np.zeros(len(anomaly_scores), dtype=int)
            classifications[alarms] = 1

            if return_anomaly_scores:
                return all_preds, anomaly_scores, classifications, thresholds, alarms
            else:
                return all_preds
        else:
            # Batch mode: use static threshold
            # Get predictions and anomaly scores
            all_preds, anomaly_scores = self._predict_and_calculate_anomaly_scores(data_loader)

            if return_anomaly_scores or plot_anomaly_scores:
                # Get static baseline threshold from EVT detector
                if isinstance(self.anomaly_detector, DriftSPOT):
                    # Validate detrended_baseline exists
                    if not hasattr(self.anomaly_detector, 'detrended_baseline') or self.anomaly_detector.detrended_baseline is None:
                        raise ValueError("DriftSPOT detector not properly fitted. detrended_baseline is missing.")
                    # DriftSPOT uses detrended baseline, so use its length
                    baseline_threshold = self.anomaly_detector.spot.evt.get_current_threshold(len(self.anomaly_detector.detrended_baseline))
                else:  # SPOT
                    baseline_threshold = self.anomaly_detector.evt.get_current_threshold(len(self.train_anomaly_scores))

                # Classify using static threshold
                classifications = (anomaly_scores > baseline_threshold).astype(int)

                if plot_anomaly_scores:
                    self.compute_and_plot_anomaly_scores(anomaly_scores, threshold=baseline_threshold)

                if return_anomaly_scores:
                    return all_preds, anomaly_scores, classifications

            return all_preds

    def compute_and_plot_anomaly_scores(
        self,
        anomaly_scores: np.ndarray,
        threshold: float = None,
        threshold_std: float = 2.0
    ):
        """
        Plot anomaly scores and print threshold/anomaly count.
        """
        if threshold is None:
            threshold = anomaly_scores.mean() + threshold_std * anomaly_scores.std()

        plt.figure(figsize=(8, 5))
        plt.hist(anomaly_scores, bins=30, alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='dashed', label=f"Threshold: {threshold:.2f}")
        plt.xlabel("Anomaly Score (Mean Absolute Error)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Anomaly Scores (Validation Set)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"Anomaly threshold: {threshold:.4f}")
        print(f"Number of anomalies above threshold: {(anomaly_scores > threshold).sum()}")

    def _predict_and_calculate_anomaly_scores(self, data_loader: DataLoader) -> Tuple[list, np.ndarray]:
        """
        Run model predictions and calculate anomaly scores for a given data_loader.
        Returns: (all_preds, anomaly_scores)
        """
        self.model.eval()
        all_preds = []
        all_anomaly_scores = []

        with torch.inference_mode():
            for batch in data_loader:
                load_seq, traffic_seq, log_seq, target_load, target_traffic, target_log = [x.to(self.device) for x in batch]
                pred_load, pred_traffic, pred_log = self.model.predict(load_seq, traffic_seq, log_seq, device=self.device)

                all_preds.append((pred_load.cpu(), pred_traffic.cpu(), pred_log.cpu()))

                # Calculate anomaly scores using shared utility function
                from utils.detection import calculate_anomaly_scores
                batch_scores = calculate_anomaly_scores(
                    pred_load, target_load,
                    pred_traffic, target_traffic,
                    pred_log, target_log
                )
                all_anomaly_scores.extend(batch_scores)

        return all_preds, np.array(all_anomaly_scores)

    def run_anomaly_detection_on_stream(
        self,
        stream_loader: DataLoader,
        plot_results: bool = True,
        update_with_alarm: bool = False
    ) -> dict:
        """
        Run anomaly detection on a stream of data.

        Args:
            stream_loader: DataLoader containing the stream data
            plot_results: Whether to plot the results
            update_with_alarm: Whether to update the model with alarm points

        Returns:
            dict containing:
                - alarms: indices of detected anomalies
                - thresholds: dynamic thresholds for each point
                - anomaly_scores: raw anomaly scores
                - predictions: list of prediction tuples (pred_load, pred_traffic, pred_log)
        """
        # Calculate anomaly scores for the stream
        all_preds, stream_scores = self._predict_and_calculate_anomaly_scores(stream_loader)

        # Run the stream through the detector
        results = self.anomaly_detector.run_stream(update_with_alarm=update_with_alarm, stream_data=stream_scores)

        if plot_results:
            self._plot_stream_results(stream_scores, results["thresholds"], results["alarms"])

        return {
            "alarms": results["alarms"],
            "thresholds": results["thresholds"],
            "anomaly_scores": stream_scores,
            "predictions": all_preds
        }

    def _plot_stream_results(
        self,
        anomaly_scores: np.ndarray,
        thresholds: list,
        alarms: list
    ):
        """
        Plot the stream results showing anomaly scores, thresholds, and detected anomalies.
        """
        plt.figure(figsize=(12, 6))

        # Plot anomaly scores
        plt.plot(anomaly_scores, label='Anomaly Scores', alpha=0.7)

        # Plot thresholds
        plt.plot(thresholds, label='Dynamic Threshold', color='red', linestyle='--', alpha=0.7)

        # Mark alarms
        if alarms:
            plt.scatter(alarms, anomaly_scores[alarms],
                       color='red', label='Detected Anomalies', zorder=5)

        plt.xlabel('Time Step')
        plt.ylabel('Anomaly Score')
        plt.title(f'Stream Anomaly Detection Results ({self.spot_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"\nAnomaly Detection Summary ({self.spot_type}):")
        print(f"Total points analyzed: {len(anomaly_scores)}")
        print(f"Number of anomalies detected: {len(alarms)}")
        if alarms:
            print(f"Anomaly indices: {alarms}")
