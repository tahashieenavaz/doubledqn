import numpy
from dataclasses import dataclass


@dataclass
class DoubleDQNBuffer:
    capacity: int
    image_size: int
    batch_size: int

    def __init__(self):
        self.states = numpy.zeros(
            (self.capacity, self.image_size, self.image_size), dtype=numpy.uint8
        )
        self.next_states = numpy.zeros(
            (self.capacity, self.image_size, self.image_size), dtype=numpy.uint8
        )
        self.actions = numpy.zeros((self.capacity, 1), dtype=numpy.long)
        self.rewards = numpy.zeros((self.capacity, 1), dtype=numpy.float32)
        self.terminations = numpy.zeros((self.capacity, 1), dtype=numpy.float32)

        self.size = 0
        self.ptr = 0

    def add(self, state, action, reward, next_state, termination) -> None:
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.reward[self.ptr] = reward
        self.termination[self.ptr] = termination
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        idx = numpy.random.permutation(self.size)[: self.batch_size]
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = self.next_states[idx]
        terminations = self.terminations[idx]
        return states, actions, rewards, next_states, terminations
