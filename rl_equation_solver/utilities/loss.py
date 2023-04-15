"""Collection of loss functions"""
import torch


class LossMixin:
    """Mixin class with collection of loss functions"""

    def huber_loss(self, x, y, delta=1.0):
        """
        Huber loss. Huber loss, also known as Smooth Mean Absolute Error, is
        a loss function used in various machine learning and optimization
        problems, particularly in regression tasks. It combines the properties
        of both Mean Squared Error (MSE) and Mean Absolute Error (MAE) loss
        functions, providing a balance between the two.

        .. math::
        :nowrap:

        L(y, f(x)) =
            0.5 * (y - f(x))^2 if |y - f(x)| <= delta \\
            delta * |y - f(x)| - 0.5 * delta^2 otherwise

        """
        return torch.nn.HuberLoss(delta=delta)(x, y)

    def smooth_l1_loss(self, x, y):
        """Smooth L1 Loss"""
        return torch.nn.SmoothL1Loss()(x, y)

    def l2_loss(self, x, y):
        """L2 Loss"""
        return torch.nn.MSELoss()(x, y)
