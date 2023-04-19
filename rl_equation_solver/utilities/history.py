"""History mixin class"""
import numpy as np
import logging


logger = logging.getLogger(__name__)


class HistoryMixin:
    """Collection of history method"""

    def __init__(self):
        self._history = {}
        self.current_episode = 0
        self.steps_done = 0

    @property
    def history(self):
        """Get training history of policy_network"""
        return self._history

    @property
    def avg_history(self):
        """Get history averaged over each episode"""
        out = {k: [] for k in self.history[0] if k not in ('state', 'approx')}
        for _, series in self.history.items():
            for k in out:
                out[k].append(np.nanmean(series[k]))
        return out

    @property
    def best_history(self):
        """Get history averaged over each episode"""
        out = {k: [] for k in self.history[0] if k not in ('approx')}
        for ep, series in self.history.items():
            for k in out:
                if k in ('state'):
                    best_i = np.argmax(self.history[ep]['reward'])
                    out[k].append(series[k][best_i])
                elif k in ('loss', 'complexity'):
                    out[k].append(np.nanmin(series[k]))
                else:
                    out[k].append(np.nanmax(series[k]))

        return out

    @property
    def end_history(self):
        """Get history averaged over each episode"""
        out = {k: [] for k in self.history[0] if k not in ('state', 'approx')}
        for _, series in self.history.items():
            for k in out:
                out[k].append(series[k][-1])
        return out

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
        logger.debug(f'\n{out}')

    def reset_history(self):
        """Clear history"""
        self._history = {}
        self.current_episode = 0
        self.steps_done = 0
