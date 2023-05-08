"""History mixin class."""
import logging
from typing import Any, Union

import numpy as np
import torch
from sympy import parse_expr, symbols
from sympy.core import Expr
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """Attribute dictionary."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the AttrDict."""
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class ProgressBar:
    """tqdm progress bar encapsulation."""

    def __init__(self, num_episodes: int, show_progress: bool):
        """Initialize the progress bar class."""
        self._pbar = None
        self.num_episodes = num_episodes
        self.show_progress = show_progress

    @property
    def pbar(self) -> Union[tqdm, None]:
        """Progress bar from tqdm."""
        if self._pbar is None and self.show_progress:
            self._pbar = tqdm(self.num_episodes)
        return self._pbar

    def clear(self) -> None:
        """Clear progress bar."""
        if self.pbar is not None:
            self.pbar.refresh()
            self.pbar.close()

    def update_desc(self, desc: str) -> None:
        """Update progress bar description."""
        if self.pbar is not None:
            self.pbar.set_description(desc)

    def update(self, count: int) -> None:
        """Update the progress bar for given number of steps."""
        if self.pbar is not None:
            self.pbar.update(count)

    def pop(self) -> None:
        """Close the progress bar."""
        if self.pbar is not None:
            self._pbar = None


class History:
    """Collection of history methods."""

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Initialize a new History."""
        self._history: list[dict] = []
        self.pbar: ProgressBar = ProgressBar(
            num_episodes=0, show_progress=False
        )
        self.reset_steps: int = 100
        self.reward: float = 0
        self.complexity: float = np.nan
        self.current_episode: int = 0
        self.steps_done: int = 0
        self.loop_step: int = 0
        self.solution_approx: Expr = symbols("0")
        self.max_solution_steps: int = 10000
        self.update_freq: int = 10
        self._state_string: Expr = symbols("0")
        self._loss = None

    @property
    def loss(self) -> torch.Tensor:
        """Get the current loss value."""
        if self._loss is None:
            loss = torch.Tensor([np.nan])
        else:
            loss = self._loss
        return loss

    @loss.setter
    def loss(self, value):
        """Set loss value."""
        self._loss = value

    @property
    def state_string(self) -> Expr:
        """Get state string representation."""
        return self._state_string

    @property
    def info(self) -> AttrDict:
        """Get info attrs."""
        return AttrDict(
            ep=self.current_episode,
            step=self.steps_done,
            loop_step=self.loop_step,
            complexity=self.complexity,
            loss=self.loss,
            reward=self.reward,
            state=self.state_string,
            approx=self.solution_approx,
        )

    @property
    def history(self) -> list[dict]:
        """Get training history of policy_network."""
        return self._history

    @history.setter
    def history(self, value: list[dict]) -> None:
        """Set training history of policy_network."""
        self._history = value

    @property
    def avg_history(self) -> dict[str, list]:
        """Get history averaged over each episode."""
        out: dict[str, list] = {
            k: [] for k in self.history[0] if k not in ("state", "approx")
        }
        for series in self.history:
            for k in out:
                if k in ("loss", "reward", "complexity"):
                    out[k].append(np.nanmean(series[k]))
                else:
                    out[k].append(series[k][-1])
        return out

    @property
    def best_history(self) -> dict[str, list]:
        """Get best loss, complexity, reward for each episode."""
        out: dict[str, list] = {
            k: [] for k in self.history[0] if k not in ("approx")
        }
        for _, series in enumerate(self.history):
            for k in out:
                if "state" in k:
                    out[k].append(series[k][-1])
                elif k in ("loss", "complexity"):
                    out[k].append(np.nanmin(series[k]))
                elif k in ("reward"):
                    out[k].append(np.nanmax(series[k]))
                else:
                    out[k].append(series[k][-1])

        return out

    @property
    def end_history(self) -> dict[str, list]:
        """Get history averaged over each episode."""
        out: dict[str, list] = {
            k: [] for k in self.history[0] if k not in ("state", "approx")
        }
        for series in self.history:
            for k in out:
                out[k].append(series[k][-1])
        return out

    @property
    def total_history(self) -> dict[str, list]:
        """Get history summed over each episode."""
        out: dict[str, list] = {
            k: [] for k in self.history[0] if k not in ("state", "approx")
        }
        for series in self.history:
            for k in out:
                if k in ("loss", "reward", "complexity"):
                    out[k].append(np.nansum(series[k]))
                else:
                    out[k].append(series[k][-1])
        return out

    def append_history(self, entry: dict) -> None:
        """Append latest step for training history of policy_network."""
        episode = self.current_episode
        if episode >= len(self._history):
            self._history.append({k: [] for k in entry.keys()})
        for k, v in entry.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().item()
            self._history[episode][k].append(v)

    def update_history(self, key: str, value: Any) -> None:
        """Update latest step for training history of policy_network."""
        if key not in self._history[-1]:
            self._history[-1][key] = [None]
        self._history[-1][key][-1] = value

    def get_non_nan(self, arr: list) -> np.ndarray:
        """Get non nan values in array."""
        out = np.array(arr)[np.where(~np.isnan(arr))[0]]
        if len(out) < 1:
            out = np.array([np.nan])
        return out

    def get_log_info(self) -> dict:
        """Get log message."""
        hist = self.get_extended_history()
        out: dict[str, str] = {}
        _out = {}
        for k, v in hist.items():
            if isinstance(v, torch.Tensor):
                _out[k] = v.cpu().item()
            else:
                _out[k] = v
        for k in ("approx",):
            _out.pop(k)
        for k, v in _out.items():
            if any(key in k for key in ("loss",)):
                tmp = self.get_non_nan(v)[-1]
                out[k] = f"{tmp:.2e}"
            else:
                out[k] = v[-1]
        for k in out:
            if "state" not in k:
                if k in ("loss", "reward", "complexity"):
                    out[k] = f"{out[k]:<9}"
                else:
                    out[k] = f"{out[k]:<4}"
            else:
                out[k] = f"{str(out[k]):<20}"
        return out

    def get_extended_history(self):
        """Get history for past two episodes if possible."""
        hist = self.history[-1].copy()
        if len(self.history) > 1:
            hist = self.history[-2].copy()
            for k, v in hist.items():
                hist[k] = np.concatenate([v, self.history[-1][k]])
        return hist

    def get_terminate_msg(self) -> str:
        """Get log message about solver termination."""
        total_reward = np.nansum(self.history[-1]["reward"])
        mean_loss = np.nanmean(self.history[-1]["loss"])
        msg = (
            f"\nEpisode: {self.current_episode}, "
            f"steps_done: {self.steps_done}. "
            f"loop_steps: {self.loop_step}, "
            f"total_reward = {total_reward:.3e}, "
            f"mean_loss = {mean_loss:.3e}, "
            f"state = {parse_expr(str(self.state_string))}."
        )
        return msg

    def write_terminate_msg(self) -> None:
        """Write terminate message to logger."""
        logger.info(self.get_terminate_msg())

    def reset_history(self) -> None:
        """Clear history."""
        self._history = []
        self.current_episode = 0
        self.steps_done = 0
        self.loop_step = 0

    def update_info(self, key: str, value: Any) -> None:
        """Update history info with given value for the given key."""
        setattr(self.info, key, value)

    # pylint: disable=invalid-unary-operand-type
    def is_constant_complexity(self) -> bool:
        """Check for constant loss over a long number of steps."""
        complexities = self.history[-1]["complexity"]
        check = (
            len(complexities) >= self.reset_steps
            and len(set(complexities[-self.reset_steps :])) <= 1
        )
        if check:
            logger.info(
                "Complexity has been constant "
                f"({list(complexities)[-1]}) for {self.reset_steps} "
                "steps. Reseting."
            )
        return check
