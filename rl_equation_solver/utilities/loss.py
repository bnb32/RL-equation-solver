"""Collection of loss functions"""
import torch


class LossMixin:
    """Mixin class with collection of loss functions"""

    def huber_loss(self, x, y, delta=1.0):
        """Huber loss"""
        return torch.nn.HuberLoss(delta=delta)(x, y)

    def smooth_l1_loss(self, x, y):
        """Smooth L1 Loss"""
        return torch.nn.SmoothL1Loss()(x, y)

    def l2_loss(self, x, y):
        """L2 Loss"""
        return torch.nn.MSELoss()(x, y)
