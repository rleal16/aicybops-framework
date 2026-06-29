import torch
import numpy as np


def calculate_anomaly_scores(
    pred_load: torch.Tensor,
    target_load: torch.Tensor,
    pred_traffic: torch.Tensor,
    target_traffic: torch.Tensor,
    pred_log: torch.Tensor,
    target_log: torch.Tensor
) -> np.ndarray:
    """MAE-based anomaly scoring: mean absolute error across all modalities per sample."""
    ae_load = torch.abs(pred_load - target_load)
    ae_traffic = torch.abs(pred_traffic - target_traffic)
    ae_log = torch.abs(pred_log - target_log)
    return torch.cat([ae_load, ae_traffic, ae_log], dim=1).mean(dim=1).cpu().numpy()
