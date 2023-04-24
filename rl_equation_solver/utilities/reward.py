"""Collection of reward functions"""
from abc import abstractmethod

import numpy as np


class RewardMixin:
    """Reward function collection"""

    @abstractmethod
    def get_solution_complexity(self, state):
        """Get the graph / expression complexity for a given state. This is
        equal to number_of_nodes + number_of_edges"""

    @abstractmethod
    def get_expression_complexity(self, expr):
        """Get the graph / expression complexity for a given expression. This
        is equal to number_of_nodes + number_of_edges"""

    @abstractmethod
    def get_solution_approx(self, state):
        """Get the approximate solution for the given state."""

    def diff_loss_reward(self, state_old, state_new):
        """
        Reward is decrease in complexity

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
        loss_old = self.get_solution_complexity(state_old)
        loss_new = self.get_solution_complexity(state_new)
        return loss_old - loss_new

    # pylint: disable=unused-argument
    def sub_loss_reward(self, state_old, state_new):
        """
        Reward is decrease in complexity

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
        loss_new = self.get_solution_complexity(state_new)
        return -1 * loss_new

    # pylint: disable=unused-argument
    def exp_loss_reward(self, state_old, state_new):
        """
        Reward is decrease in complexity

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
        loss_new = self.get_solution_complexity(state_new)
        return np.exp(-loss_new)

    # pylint: disable=unused-argument
    def inv_loss_reward(self, state_old, state_new):
        """
        Reward is decrease in complexity

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
        loss_new = self.get_solution_complexity(state_new)
        return 1 / (1 + loss_new)
