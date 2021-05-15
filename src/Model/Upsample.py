from __future__ import division
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """nn.Upsample is deprecated"""

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
