"""MlpPolicy"""
import torch

from rl_equation_solver.policy.base import BasePolicy


class MlpPolicy(BasePolicy):
    """MlpPolicy with target and policy networks"""

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
        with torch.no_grad():
            out = batch.rewards
            out += self.gamma * torch.mul(1 - batch.dones, self.compute_next_Q(batch))
        return out

    def compute_next_Q(self, batch):
        """
        Compute :math:`max_{a} Q(s_{t+1}, a)` for all next states. Expected
        values for next_states are computed based on the "older" target_net;
        selecting their best reward].
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values = self.target_network(batch.next_states).max(1)[0]

        return next_state_values

    def compute_Q(self, batch):
        """
        Compute :math:`Q(s_t, a)`. These are the actions which would've
        been taken for each batch state according to policy_net
        """
        return self.policy_network(batch.states).gather(1, batch.actions)

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
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip)
        self.optimizer.step()
