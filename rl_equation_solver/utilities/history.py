"""History mixin class"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HistoryMixin:
    """Collection of history method"""

    def __init__(self):
        self._history = {}

    @property
    def history(self):
        """Get training history of policy_network"""
        return self._history

    @property
    def avg_history(self):
        """Get history averaged over each episode"""
        out = {k: [] for k in self.history[0] if k != 'state'}
        for _, series in self.history:
            for k in out:
                out[k].append(np.nanmean(series[k]))

    @history.setter
    def history(self, value):
        """Set training history of policy_network"""
        self._history = value

    def append_history(self, entry):
        """Append latest step for training history of policy_network"""
        episode = entry['ep']
        if episode not in self._history:
            self._history[episode] = {k: [] for k in entry.keys()}
        for k, v in entry.items():
            self._history[episode][k].append(v)

    def update_history(self, key, value):
        """Update latest step for training history of policy_network"""
        episode = list(self.history.keys())[-1]
        self._history[episode][key][-1] = value

    def log_info(self):
        """Write info to logger"""
        out = self.history[list(self.history.keys())[-1]]
        out = {k: v[-1] for k, v, in out.items()}
        out['reward'] = '{:.3e}'.format(out['reward'])
        out['loss'] = '{:.3e}'.format(out['loss'])
        logger.info(out)
