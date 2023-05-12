"""Search for optimal hyperparameters for a given environment and model."""
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Union

import numpy as np

from rl_equation_solver.environment.algebraic import Env

logger = logging.getLogger(__name__)


class Tuner:
    """Parameter searching class."""

    def __init__(
        self,
        params: dict,
        agent,
        env_kwargs: Union[dict, None] = None,
        run_number: int = 10,
        max_workers: Union[int, None] = None,
    ):
        """Initialize the tuner with the given parameters.

        Parameters
        ----------
        params : dict
            dictionary of parameters to search, with keys as the parameter
            names and values as arrays/lists of values
        env_kwargs : dict
            Arguments for environment to use for parameter search
        agent : BaseAgent
            Agent class to use for parameter search
        run_number : int
            Number of runs to use for a single parameter set
        max_workers : int
            Number of workers to use for running parameters search in
            parallel.
        """
        self.histories: dict = {}
        self.configs: dict = {}
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.agent = agent
        self.run_number = run_number
        self.max_workers = max_workers

        combos = list(itertools.product(*list(params.values())))
        for i, combo in enumerate(combos):
            self.configs[i] = dict(zip(list(params.keys()), combo))

        for i in range(len(self.configs)):
            self.histories[i] = {}

    def find_solution(self, i: int, j: int) -> None:
        """Find the solution for the parameter set specified by i and j."""
        config = self.configs[i]
        env_kwargs = deepcopy(self.env_kwargs)
        env_kwargs["config"] = config
        env = Env(**env_kwargs)
        agent = self.agent(env, config=config)
        agent.train(1, progress_bar=False)

        for k, v in env.history[-1].items():
            if k not in ("state", "approx"):
                if k not in self.histories[i]:
                    self.histories[i][k] = [np.nanmean(v)]
                else:
                    self.histories[i][k].append(np.nanmean(v))

    def run(self, optimize_key: str = "step", get_min: bool = True) -> dict:
        """Find the solution for each parameter set in the configs dict."""
        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            count = 0
            for i in range(len(self.configs)):
                for j in range(self.run_number):
                    future = pool.submit(self.find_solution, i, j)
                    futures[future] = [count, i, j]
                    count += 1

            for _, future in enumerate(as_completed(futures)):
                try:
                    _ = future.result()
                except Exception as e:
                    msg = (
                        f"future {futures[future][0]} failed: "
                        f"({futures[future][1:]})"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg) from e
                logger.info(
                    f"future {futures[future][0] + 1} of {len(futures)} "
                    "completed."
                )

        best = self.get_best_config(optimize_key, get_min=get_min)
        return best

    def get_best_config(self, optimize_key: str, get_min: bool = True) -> dict:
        """Get the config which minimizes optimize_key."""
        avgs = {}
        for i in self.histories:
            avgs[i] = np.mean(np.nanmean(self.histories[i][optimize_key]))
        if get_min:
            best = self.configs[np.argmin(list(avgs.values()))]
        else:
            best = self.configs[np.argmax(list(avgs.values()))]
        logger.info(f"Best config: {best}")
        return best
