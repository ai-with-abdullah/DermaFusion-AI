import torch
import torch.nn.functional as F
import numpy as np

class ModelEMA:
    """
    Exponential Moving Average of model weights AND buffers.

    Tracks both named_parameters (learnable weights) and named_buffers
    (BatchNorm running_mean / running_var / num_batches_tracked) so that
    EMA validation uses fully consistent model statistics.

    Fix Bug #1: Removed the dead `self.module = type(model)(...)` line that
                would crash because DualBranchFusionClassifier has no .config.
    Fix Bug #2: register() + update() now also iterate named_buffers(), so
                BatchNorm running stats are EMA-averaged (previously missed).
    """
    def __init__(self, model, decay=0.9999, device=None):
        # EMA weights stored in self.shadow dict — no model copy needed
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.device = device
        self.register(model)

    def register(self, model):
        """Register all trainable parameters AND buffers (BN stats)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
        # Also register buffers (BatchNorm running_mean / running_var)
        for name, buf in model.named_buffers():
            if buf is not None:
                self.shadow[f'__buf__{name}'] = buf.data.clone().to(self.device)

    def update(self, model):
        """EMA update for parameters and buffers."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        for name, buf in model.named_buffers():
            if buf is not None:
                key = f'__buf__{name}'
                if key in self.shadow:
                    new_average = (1.0 - self.decay) * buf.data + self.decay * self.shadow[key]
                    self.shadow[key] = new_average.clone()

    def apply_shadow(self, model):
        """Swap in EMA weights+buffers for validation. Call restore() after."""
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
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_max = -np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """Saves model when validation metric increases."""
        if self.verbose:
            self.trace_func(f'Validation metric increased ({self.val_max:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_max = val_metric
