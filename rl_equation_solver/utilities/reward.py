"""Collection of reward functions"""
from abc import abstractmethod
import numpy as np


class RewardMixin:
    """Reward function collection"""

    @abstractmethod
    def find_loss(self, state):
        """Get loss for a given state"""

    def loss_diff_reward(self, state_old, state_new):
        """
        Reward is decrease in loss

        Parameters
        ----------
        state_old : str
            String representation of last state
        state_new : str
            String representation of new state

        Returns
        -------
        reward : int
            Difference between loss for state_new and state_old
        """
        loss_old = self.find_loss(state_old)
        loss_new = self.find_loss(state_new)
        return loss_old - loss_new

    # pylint: disable=unused-argument
    def sub_loss_reward(self, state_old, state_new):
        """
        Reward is decrease in loss

        Parameters
        ----------
        state_old : str
            String representation of last state
        state_new : str
            String representation of new state

        Returns
        -------
        reward : int
            Difference between loss for state_new and state_old
        """
        loss_new = self.find_loss(state_new)
        return -1 * loss_new

    # pylint: disable=unused-argument
    def exp_loss_reward(self, state_old, state_new):
        """
        Reward is decrease in loss

        Parameters
        ----------
        state_old : str
            String representation of last state
        state_new : str
            String representation of new state

        Returns
        -------
        reward : int
            Difference between loss for state_new and state_old
        """
        loss_new = self.find_loss(state_new)
        return np.exp(-loss_new)

    # pylint: disable=unused-argument
    def inv_loss_reward(self, state_old, state_new):
        """
        Reward is decrease in loss

        Parameters
        ----------
        state_old : str
            String representation of last state
        state_new : str
            String representation of new state

        Returns
        -------
        reward : int
            Difference between loss for state_new and state_old
        """
        loss_new = self.find_loss(state_new)
        return 1 / (1 + loss_new)