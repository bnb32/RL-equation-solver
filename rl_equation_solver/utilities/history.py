"""History mixin class"""
import numpy as np
import logging
import torch
from sympy import parse_expr


logger = logging.getLogger(__name__)


class AttrDict(dict):
    """Attribute dictionary"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# pylint: disable=unused-argument
class Info:
    """Shared info class"""

    _shared_info = {}

    def __init__(self, *args, **kwargs):
        self.state_string = None
        self.next_state_string = None
        self.loss = np.nan
        self.reward = None
        self.complexity = None
        self.current_episode = 0
        self.steps_done = 0
        self.loop_step = 0
        self.solution_approx = None
        self.reset_step = False
        self.max_solution_steps = None

    def __new__(cls, *args, **kwargs):
        """Singleton constructor"""
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_info
        return obj

    @property
    def info(self) -> AttrDict:
        """Get info attrs"""
        return AttrDict(ep=self.current_episode,
                        step=self.steps_done,
                        loop_step=self.loop_step,
                        complexity=self.complexity,
                        loss=self.loss,
                        reward=self.reward,
                        previous_state=self.state_string,
                        state=self.next_state_string,
                        approx=self.solution_approx,
                        reset_step=self.reset_step)


class History(Info):
    """Collection of history methods"""

    _shared_history = {}

    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self._history = []

    def __new__(cls, *args, **kwargs):
        """Singleton constructor"""
        obj = super(History, cls).__new__(cls)
        obj.__dict__ = cls._shared_history
        return obj

    @property
    def history(self) -> list[dict]:
        """Get training history of policy_network"""
        return self._history

    @property
    def avg_history(self) -> dict[int, list]:
        """Get history averaged over each episode"""
        out = {k: [] for k in self.history[0] if k not in ('state', 'approx')}
        for series in self.history:
            for k in out:
                if k in ('loss', 'reward', 'complexity'):
                    out[k].append(np.nanmean(series[k]))
                else:
                    out[k].append(series[-1])
        return out

    @property
    def best_history(self) -> dict[int, list]:
        """Get best loss, complexity, reward for each episode"""
        out = {k: [] for k in self.history[0] if k not in ('approx')}
        for _, series in enumerate(self.history):
            for k in out:
                if 'state' in k:
                    out[k].append(series[k][-1])
                elif k in ('loss', 'complexity'):
                    out[k].append(np.nanmin(series[k]))
                elif k in ('reward'):
                    out[k].append(np.nanmax(series[k]))
                else:
                    out[k].append(series[k][-1])

        return out

    @property
    def end_history(self) -> dict[int, list]:
        """Get history averaged over each episode"""
        out = {k: [] for k in self.history[0] if k not in ('state', 'approx')}
        for series in self.history:
            for k in out:
                out[k].append(series[k][-1])
        return out

    @property
    def total_history(self) -> dict[int, list]:
        """Get history summed over each episode"""
        out = {k: [] for k in self.history[0] if k not in ('state', 'approx')}
        for series in self.history:
            for k in out:
                if k in ('loss', 'reward', 'complexity'):
                    out[k].append(np.nansum(series[k]))
                else:
                    out[k].append(series[k][-1])
        return out

    @history.setter
    def history(self, value):
        """Set training history of policy_network"""
        self._history = value

    def append_history(self, entry):
        """Append latest step for training history of policy_network"""
        episode = entry['ep']
        if episode >= len(self._history):
            self._history.append({k: [] for k in entry.keys()})
        for k, v in entry.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().item()
            self._history[episode][k].append(v)

    def update_history(self, key, value):
        """Update latest step for training history of policy_network"""
        self._history[-1][key][-1] = value

    def get_log_info(self) -> dict:
        """Get log message"""
        out = self.history[-1].copy()
        out = {k: v[-1] for k, v, in out.items()}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu().item()
        for k in ('approx', 'reset_step'):
            out.pop(k)
        for k in out:
            if 'state' not in k:
                if k in ('loss', 'reward', 'complexity'):
                    out[k] = '{:<9}'.format('{:.2e}'.format(out[k]))
                else:
                    out[k] = '{:<4}'.format(out[k])
            else:
                out[k] = '{:<20}'.format(str(parse_expr(str(out[k]))))
        return out

    def get_terminate_msg(self) -> str:
        """Get log message about solver termination
        """
        total_reward = np.nansum(self.history[-1]['reward'])
        mean_loss = np.nanmean(self.history[-1]['loss'])
        msg = (f"\nEpisode: {self.current_episode}, "
               f"steps_done: {self.steps_done}. "
               f"loop_steps: {self.loop_step}, "
               f"total_reward = {total_reward:.3e}, "
               f"mean_loss = {mean_loss:.3e}, "
               f"state = {parse_expr(str(self.state_string))}.")
        return msg

    def write_terminate_msg(self):
        """Write terminate message to logger"""
        logger.info(self.get_terminate_msg())

    def reset_history(self):
        """Clear history"""
        self._history = []
        self.current_episode = 0
        self.steps_done = 0
        self.loop_step = 0
        self.reset_step = False

    def update_info(self, key, value):
        """Update history info with given value for the given key"""
        setattr(self.info, key, value)
