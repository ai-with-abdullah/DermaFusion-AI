import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optional: ensure that PyTorch operations are deterministic if possible
    # torch.use_deterministic_algorithms(True)
    
    print(f"Global seed set to: {seed}")
