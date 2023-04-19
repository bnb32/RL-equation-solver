"""Policy module"""
import torch
from torch import optim
import random
import math

random.seed(42)


# pylint: disable=not-callable
class MlpPolicy:
    """MlpPolicy with target and policy networks"""

    def __init__(self, env):
        """
        Parameters
        ----------
        env : Object
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        """
        self.env = env
        self.optimizer = None
        self.policy_network = None
        self.target_network = None
        self.gamma = None
        self.learning_rate = None
        self.tau = None
        self.grad_clip = None
        self.eps_end = None
        self.eps_start = None
        self.eps_decay = None
        self.device = None
        self.eps_decay_steps = None
        self.batch_size = None

    @property
    def steps_done(self):
        """Get total number of steps done across all episodes"""
        return self.env.steps_done

    @steps_done.setter
    def steps_done(self, value):
        """Set total number of steps done across all episodes"""
        self.env.steps_done = value

    def _get_eps_decay(self):
        """Get epsilon decay for current number of steps"""
        decay = 0
        if self.eps_decay is None:
            decay = self.eps_start - self.eps_end
            decay *= math.exp(-1. * self.steps_done / self.eps_decay_steps)
        return decay

    def init_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = optim.AdamW(self.policy_network.parameters(),
                                     lr=self.learning_rate, amsgrad=True)

    def choose_random_action(self):
        """Choose random action rather than the optimal action"""
        return torch.tensor([[self.env.action_space.sample()]],
                            device=self.device, dtype=torch.long)

    def choose_optimal_action(self, state):
        """
        Choose action with max expected reward :math:`:= max_a Q(s, a)`

        max(1) will return largest column value of each row. second column on
        max result is index of where max element was found so we pick action
        with the larger expected reward.
        """
        with torch.no_grad():
            return self.policy_network(state).max(1)[1].view(1, 1)

    def compute_expected_Q(self, batch):
        r"""
        Compute the expected Q values according to the Bellman optimality
        equation :math:`Q(s, a) = E(R_{s + 1} + \gamma *
        max_{a^{'}} Q(s^{'}, a^{'}))`
        """
        return batch.reward_batch + (self.gamma * self.compute_V(batch))

    def compute_V(self, batch):
        """
        Compute :math:`V(s_{t+1})` for all next states. Expected values
        for non_final_next_states are computed based on the "older"
        target_net; selecting their best reward with max(1)[0].
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[batch.non_final_mask] = \
                self.target_network(batch.non_final_next_states).max(1)[0]

        return next_state_values

    def compute_Q(self, batch):
        """
        Compute :math:`Q(s_t, a)`. These are the actions which would've
        been taken for each batch state according to policy_net
        """
        return self.policy_network(batch.state_batch) \
            .gather(1, batch.action_batch)

    def update_networks(self):
        r"""
        Soft update of the target network's weights :math:`\theta^{'}
        \leftarrow \tau \theta + (1 - \tau) \theta^{'}`
        policy_network.state_dict() returns the parameters of the policy
        network target_network.load_state_dict() loads these parameters into
        the target network.
        """
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.policy_network.state_dict()
        for key in policy_net_state_dict:
            policy = policy_net_state_dict[key]
            target = target_net_state_dict[key]
            value = target + self.tau * (policy - target)
            target_net_state_dict[key] = value
        self.target_network.load_state_dict(target_net_state_dict)

    def optimize_model(self, loss=None):
        """
        Perform one step of the optimization (on the policy network).
        """
        if loss is None:
            return

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(),
                                        self.grad_clip)
        self.optimizer.step()

    def choose_action(self, state, training=False):
        """
        Choose action based on given state. Either choose optimal action or
        random action depending on training step.
        """
        random_float = random.random()
        epsilon_threshold = self.eps_end + self._get_eps_decay()

        if not training:
            epsilon_threshold = self.eps_end

        if random_float > epsilon_threshold:
            return self.choose_optimal_action(state)
        else:
            return self.choose_random_action()
