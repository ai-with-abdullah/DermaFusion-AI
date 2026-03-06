import torch
import torch.nn.functional as F
import numpy as np


class ModelEMA:
    """
    Exponential Moving Average of model weights AND buffers.

    Tracks both named_parameters (learnable weights) and named_buffers
    (BatchNorm running_mean / running_var / num_batches_tracked) so that
    EMA validation uses fully consistent model statistics.

    UPGRADE (Fix #8): EMA decay now uses a warmup schedule:
        decay = min(config_decay, (1 + step) / (10 + step))
    This prevents the shadow from being stuck near pretrained weights in
    early epochs when gradient updates are large and noisy (batch_size=4).
    After ~10K steps the formula saturates to config_decay (e.g. 0.9998).

    FIXED (DataParallel): All methods auto-unwrap nn.DataParallel so they
    work correctly whether you pass model.module or the wrapped model.
    """
    def __init__(self, model, decay=0.9998, device=None):
        self.config_decay = decay   # Target long-run decay
        self.decay  = 0.0           # Starts at 0, warms up
        self.shadow = {}
        self.backup = {}
        self.device = device
        self._step  = 0
        self.register(model)

    @staticmethod
    def _unwrap(model):
        """Strip DataParallel wrapper so parameter names are consistent."""
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def register(self, model):
        """Register all trainable parameters AND buffers (BN stats)."""
        model = self._unwrap(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
        for name, buf in model.named_buffers():
            if buf is not None:
                self.shadow[f'__buf__{name}'] = buf.data.clone().to(self.device)

    def _update_decay(self):
        """Warmup schedule: ramp from 0 → config_decay over ~10K steps."""
        self._step += 1
        # Standard timm EMA warmup formula
        self.decay = min(self.config_decay, (1.0 + self._step) / (10.0 + self._step))

    def update(self, model):
        """EMA update with warmup decay for parameters and buffers."""
        model = self._unwrap(model)
        self._update_decay()
        d = self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"EMA: unexpected param '{name}' — was model changed after register()?"
                new_average = (1.0 - d) * param.data + d * self.shadow[name]
                self.shadow[name] = new_average.clone()
        for name, buf in model.named_buffers():
            if buf is not None:
                key = f'__buf__{name}'
                if key in self.shadow:
                    new_average = (1.0 - d) * buf.data + d * self.shadow[key]
                    self.shadow[key] = new_average.clone()

    def apply_shadow(self, model):
        """Swap in EMA weights+buffers for validation. Call restore() after."""
        model = self._unwrap(model)
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
        for name, buf in model.named_buffers():
            if buf is not None:
                key = f'__buf__{name}'
                if key in self.shadow:
                    self.backup[key] = buf.data.clone()
                    buf.data = self.shadow[key]

    def restore(self, model):
        """Restore original training weights and buffers after EMA validation."""
        model = self._unwrap(model)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        for name, buf in model.named_buffers():
            if buf is not None:
                key = f'__buf__{name}'
                if key in self.backup:
                    buf.data = self.backup[key]
        self.backup = {}


def apply_mask(image, mask):
    """
    Applies a binary predicted mask to the original image to generate a lesion-focused image.
    image: (B, 3, H, W) normalized tensor
    mask: (B, 1, H, W) logits from UNet
    """
    probs = torch.sigmoid(mask)
    binary_mask = (probs > 0.5).float()
    binary_mask_3c = binary_mask.repeat(1, 3, 1, 1)
    masked_image = image * binary_mask_3c
    return masked_image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stops training if validation metric doesn't improve after 'patience' epochs.

    FIXED (Fix #2 / Upgrade #1): Now saves EMA shadow weights when an EMA object is
    provided. Previously, the checkpoint contained raw training weights but metrics
    were measured on EMA weights — causing a 1–5% AUC gap between training logs and
    final evaluation. When ema is passed, best_dual_branch_fusion.pth will contain
    the EMA weights that actually produced the metric.
    """
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience   = patience
        self.verbose    = verbose
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.val_max    = -np.inf
        self.delta      = delta
        self.path       = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model, ema=None):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ema)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, ema)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, ema=None):
        """
        Saves the best model checkpoint.

        If an EMA object is provided, saves the EMA shadow weights (the weights
        that were used for validation and produced val_metric). Otherwise falls
        back to saving raw model weights.
        """
        if self.verbose:
            self.trace_func(
                f'Validation metric improved ({self.val_max:.6f} --> {val_metric:.6f}). '
                + ('Saving EMA weights ...' if ema is not None else 'Saving model weights ...')
            )

        if ema is not None:
            # Build a proper state_dict from EMA shadow (excluding buffer keys)
            ema_state = {
                k: v for k, v in ema.shadow.items()
                if not k.startswith('__buf__')
            }
            torch.save(ema_state, self.path)
        else:
            torch.save(model.state_dict(), self.path)

        self.val_max = val_metric
