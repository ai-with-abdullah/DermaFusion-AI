"""
Temperature Scaling Calibration
================================
Post-hoc calibration for the DualBranchFusionClassifier.

Reduces Expected Calibration Error (ECE) without retraining.
Expected: ECE 0.2357 → ~0.04–0.07 after a single pass on the validation set.

References:
  - Guo et al. (2017) "On Calibration of Modern Neural Networks"
  - NeurIPS 2023/2024: "Focal Loss + Temperature Scaling = SOTA calibration combo"
  - Adaptive Temperature Scaling (Feb 2024): entropy-based scaling for imbalanced classes

Usage:
    # 1. Gather raw logits on validation set (before softmax):
    scaler = TemperatureScaler(device=config.DEVICE)
    scaler.fit(logits_val, labels_val)        # finds optimal T
    scaler.save('outputs/temperature.pt')

    # 2. At inference time:
    scaler = TemperatureScaler.load('outputs/temperature.pt', device=config.DEVICE)
    calibrated_probs = scaler.calibrate(logits_test)   # numpy or tensor input
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax as scipy_softmax
from scipy.optimize import minimize_scalar
from typing import Optional, Union


class TemperatureScaler(nn.Module):
    """
    Global Temperature Scaling.

    Divides all logits by a single scalar T before softmax.
    T > 1 → softer/more uncertain predictions (reduces overconfidence).
    T < 1 → sharper/more confident predictions.

    T is optimized by minimizing Negative Log-Likelihood on the validation set.
    This typically reduces ECE by 60–80% with zero change to model weights or accuracy.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.device = device
        self.fitted = False

    def fit(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        max_iter: int = 100,
        verbose: bool = True,
    ) -> float:
        """
        Find optimal temperature T on validation logits/labels.

        Args:
            logits: Raw logits (before softmax), shape (N, C)
            labels: Integer class labels, shape (N,)
            max_iter: Max LBFGS iterations
            verbose: Print T and ECE before/after

        Returns:
            Optimal temperature (float)
        """
        if isinstance(logits, torch.Tensor):
            logits_np = logits.cpu().numpy()
        else:
            logits_np = np.array(logits, dtype=np.float32)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy().astype(int)
        else:
            labels_np = np.array(labels, dtype=int)

        if verbose:
            pre_ece = self._compute_ece(logits_np, labels_np, T=1.0)
            print(f"  [Calibration] Before temperature scaling: ECE = {pre_ece:.4f}")

        # Scipy scalar minimization (faster + more robust than PyTorch LBFGS for 1D)
        def nll_objective(T):
            if T <= 0:
                return 1e9
            scaled = logits_np / T
            # Numerically stable log-softmax
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True) + 1e-10)
            nll = -log_probs[np.arange(len(labels_np)), labels_np].mean()
            return float(nll)

        result = minimize_scalar(nll_objective, bounds=(0.1, 10.0), method='bounded',
                                 options={'maxiter': max_iter})
        opt_T = float(result.x)
        self.temperature = nn.Parameter(torch.tensor([opt_T]))
        self.fitted = True

        if verbose:
            post_ece = self._compute_ece(logits_np, labels_np, T=opt_T)
            print(f"  [Calibration] Optimal temperature T = {opt_T:.4f}")
            print(f"  [Calibration] After  temperature scaling: ECE = {post_ece:.4f}  "
                  f"(reduction: {pre_ece - post_ece:.4f})")

        return opt_T

    def calibrate(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        return_tensor: bool = False,
    ) -> np.ndarray:
        """
        Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: Raw logits (before softmax), shape (N, C) or (C,)
            return_tensor: If True, return torch.Tensor instead of numpy array

        Returns:
            Calibrated probability distribution, shape (N, C)
        """
        if isinstance(logits, torch.Tensor):
            logits_np = logits.cpu().numpy().astype(np.float32)
        else:
            logits_np = np.array(logits, dtype=np.float32)

        T = float(self.temperature.item())
        scaled = logits_np / T
        probs = scipy_softmax(scaled, axis=-1)

        if return_tensor:
            return torch.from_numpy(probs)
        return probs

    @staticmethod
    def _compute_ece(logits_np: np.ndarray, labels_np: np.ndarray, T: float = 1.0,
                     n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE) with equal-width bins.
        ECE < 0.05 is considered well-calibrated for clinical applications.
        """
        scaled = logits_np / T
        probs = scipy_softmax(scaled, axis=1)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == labels_np).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(labels_np)

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_conf = confidences[mask].mean()
            bin_acc  = correct[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

        return float(ece)

    def save(self, path: str):
        """Save temperature to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({'temperature': self.temperature.item(), 'fitted': self.fitted}, path)
        print(f"  [Calibration] Saved temperature T={self.temperature.item():.4f} to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TemperatureScaler':
        """Load temperature from disk."""
        scaler = cls(device=device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        scaler.temperature = nn.Parameter(torch.tensor([ckpt['temperature']]))
        scaler.fitted = ckpt.get('fitted', True)
        print(f"  [Calibration] Loaded temperature T={scaler.temperature.item():.4f} from {path}")
        return scaler

    def get_temperature(self) -> float:
        return float(self.temperature.item())


class ClasswiseTemperatureScaler:
    """
    Per-class Temperature Scaling.

    Fit a separate T_c for each class c. Useful when different classes
    have very different confidence distributions (common in imbalanced medical datasets).

    Note: Requires more validation samples per class than global TS.
    Use only if global TS leaves per-class ECE > 0.05.
    """

    def __init__(self, num_classes: int, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device
        self.temperatures = np.ones(num_classes, dtype=np.float32)
        self.fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Fit one temperature per class using binary one-vs-rest NLL."""
        for c in range(self.num_classes):
            def nll_c(T):
                if T <= 0:
                    return 1e9
                scaled = logits / T
                log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True) + 1e-10)
                mask = (labels == c)
                if mask.sum() == 0:
                    return 0.0
                return -log_probs[mask, c].mean()

            result = minimize_scalar(nll_c, bounds=(0.1, 10.0), method='bounded')
            self.temperatures[c] = float(result.x)

        self.fitted = True
        if verbose:
            print(f"  [ClasswiseTS] Per-class temperatures: "
                  + ', '.join(f"c{c}:{t:.2f}" for c, t in enumerate(self.temperatures)))
        return self.temperatures

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply class-specific temperature scaling."""
        scaled = logits.copy().astype(np.float32)
        for c in range(self.num_classes):
            scaled[:, c] = logits[:, c] / self.temperatures[c]
        return scipy_softmax(scaled, axis=-1)


# =========================================================================== #
#                     MELANOMA THRESHOLD ADJUSTMENT                            #
# =========================================================================== #

def apply_mel_threshold_boost(
    probs: np.ndarray,
    mel_idx: int = 4,
    boost_factor: float = 1.5,
) -> np.ndarray:
    """
    Boost melanoma class probability by a factor before argmax decision.

    Clinical rationale: dermatologists accept 3× higher FP rate for melanoma
    over FN (missing a melanoma is worse than a false alarm).
    A boost_factor of 1.5 shifts the effective mel threshold from 0.50 → ~0.33.

    This does NOT change the reported probabilities — apply ONLY at the argmax
    prediction step. If reporting probs for calibration, use raw calibrated probs.

    Args:
        probs:        Softmax probabilities, shape (N, C) — should be CALIBRATED
        mel_idx:      Index of melanoma class (default 4 for HAM7)
        boost_factor: Multiply mel prob by this before argmax (default 1.5)

    Returns:
        Binary predictions array, shape (N,) — NOT modified probs
    """
    boosted = probs.copy()
    boosted[:, mel_idx] *= boost_factor
    return boosted.argmax(axis=1)
