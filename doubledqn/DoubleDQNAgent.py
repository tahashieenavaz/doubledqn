import torch
from dataclasses import dataclass
from typing import NewType
from atarihelpers import make_environment
from .DoubleDQNNetwork import DoubleDQNNetwork

LossValue = NewType("LossValue", float)


@dataclass
class DoubleDQNAgent:
    environment: str
    timesteps: int = 10_000_000
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    lr: float = 0.00025 / 4
    training_starts: int = 80_000
    train_every: int = 4
    convolution_activation_function = torch.nn.GELU
    stream_activation_function = torch.nn.GELU
    hidden_dimension = 512
    gamma: float = 0.99

    def __init__(self):
        self.t = 0
        self.environment_identifier = self.environment
        self.environment = make_environment(self.environment_identifier)
        self.action_dimension = self.environment.action_space.n
        self.network = DoubleDQNNetwork(
            action_dimension=self.action_dimension,
            hidden_dimension=self.hidden_dimension,
            convolution_activation_function=self.convolution_activation_function,
            stream_activation_function=self.stream_activation_function,
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    @property
    def learning_starts(self):
        return self.training_starts

    @property
    def epsilon(self) -> float:
        decay_factor_power = self.t / self.timesteps
        decay_fraction = self.final_epsilon / self.initial_epsilon
        decay_factor = decay_fraction**decay_factor_power
        _epsilon = self.initial_epsilon * decay_factor
        return min(self.final_epsilon, _epsilon)

    def train(self) -> LossValue:
        if self.t < self.learning_starts:
            return 0.0

        if self.t % self.train_every != 0:
            return 0.0

        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(self, states, actions, rewards, next_states, terminations):
        q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_target = self.target(
            next_states=next_states, terminations=terminations, rewards=rewards
        )
        return torch.nn.functional.huber_loss(q, q_target)

    @torch.no_grad()
    def target(self, next_states, terminations, rewards):
        next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
        next_q = (
            self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        )
        return rewards + (1 - terminations) * next_q * self.gamma

    def tick(self):
        self.t += 1
