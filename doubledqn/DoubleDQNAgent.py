import torch
from dataclasses import dataclass


@dataclass
class DoubleDQNAgent:
    environment: str
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    lr: float = 0.00025 / 4
    convolution_activation_function = torch.nn.GELU
    stream_activation_function = torch.nn.GELU

    def __init__(self):
        self.network = DoubleDQNNetwork()
        self.optimizer = torch.optim.Adam(self.network.parameters())

    def train(self):
        pass

    def loss(self):
        pass
