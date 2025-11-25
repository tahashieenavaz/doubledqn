import torch
from dataclasses import dataclass
from typing import Type


@dataclass
class DoubleDQNNetwork(torch.nn.Module):
    action_dimension: int
    hidden_dimension: int
    stream_activation_function: Type[torch.nn.Module]
    convolution_activation_function: Type[torch.nn.Module]

    def __init__(self):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            self.convolution_activation_function(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            self.convolution_activation_function(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            self.convolution_activation_function(),
            torch.nn.Flatten(),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(3136, self.hidden_dimension),
            self.stream_activation_function(),
            torch.nn.Linear(self.hidden_dimension, 1),
        )
        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(3136, self.hidden_dimension),
            self.stream_activation_function(),
            torch.nn.Linear(self.hidden_dimension, self.action_dimension),
        )

    def forward(self, state: torch.Tensor):
        features = self.phi(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(keepdim=True, dim=1)
