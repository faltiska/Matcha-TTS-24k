import torch
import torch.nn as nn

class OdeSolverWrapper(nn.Module):
    def __init__(self, estimator, mask, mu):
        super().__init__()
        self.estimator = estimator
        self.mask = mask
        self.mu = mu

    def forward(self, t, x):
        """
        torchdiffeq calls this as forward(t, x).
        We map it to your original signature: estimator(x, mask, mu, t)
        """
        return self.estimator(x, self.mask, self.mu, t)