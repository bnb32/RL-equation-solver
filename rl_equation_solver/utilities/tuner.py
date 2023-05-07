"""Search for optimal hyperparameters for a given environment and model."""
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Union

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
        self.solution_steps: dict = {}
        self.configs: dict = {}
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.agent = agent
        self.run_number = run_number
        self.max_workers = max_workers

        combos = list(itertools.product(*list(params.values())))
        for i, combo in enumerate(combos):
            self.configs[i] = dict(zip(list(params.keys()), combo))

    def find_solution(self, i: int, j: int) -> None:
        """Find the solution for the parameter set specified by i and j."""
        config = self.configs[i]
        env_kwargs = deepcopy(self.env_kwargs)
        env_kwargs["config"] = config
        env = Env(**env_kwargs)
        agent = self.agent(env, config=config)
        agent.train(1, progress_bar=False)
        self.solution_steps[i][j] = env.history[-1]["step"][-1]

    def run(self) -> dict:
        """Find the solution for each parameter set in the configs dict."""
        for i in range(len(self.configs)):
            self.solution_steps[i] = [0] * self.run_number

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
                    f"future {futures[future][0]} of {len(futures)} completed."
                )

        return self.solution_steps
