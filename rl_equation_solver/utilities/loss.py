"""Collection of loss functions."""
import torch


class LossMixin:
    """Mixin class with collection of loss functions."""

    def huber_loss(
        self, x: torch.Tensor, y: torch.Tensor, delta: float = 1.0
    ) -> torch.Tensor:
        r"""Huber loss. Huber loss, also known as Smooth Mean Absolute Error,
        is a loss function used in various machine learning and optimization
        problems, particularly in regression tasks. It combines the properties
        of both Mean Squared Error (MSE) and Mean Absolute Error (MAE) loss
        functions, providing a balance between the two.

        .. math::
            :nowrap:

            \[
            L(y, f(x)) =
              \begin{cases}
                \begin{split}
                  \frac{1}{2} (y - f(x))^2, & \text{ if } |y - f(x)| \leq
                  \delta \\
                  \delta |y - f(x)| - \frac{1}{2} \delta^2, & \text{ otherwise}
                \end{split}
              \end{cases}
            \]
        """
        return torch.nn.HuberLoss(delta=delta)(x, y)

    def smooth_l1_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Smooth L1 Loss."""
        return torch.nn.SmoothL1Loss()(x, y)

    def l2_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L2 Loss."""
        return torch.nn.MSELoss()(x, y)
