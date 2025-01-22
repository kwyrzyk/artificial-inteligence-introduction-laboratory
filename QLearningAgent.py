import numpy as np
import random
from typing import Protocol, List, Tuple
from dataclasses import dataclass

class Environment(Protocol):
    def reset(self) -> int: ...
    def step(self, action: int) -> Tuple[int, int, bool]: ...

class SelectStrategy(Protocol):
    def select(self, q_table: np.ndarray, state: int, actions_num: int) -> int: ...

class EpsilonGreedyStrategy(SelectStrategy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select(self, q_table: np.ndarray, state: int, actions_num: int) -> int:
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, actions_num - 1)
        else:
            return np.argmax(q_table[state])
    
    def __str__(self) -> str:
        return 'Epsilon Greedy Strategy'

class BoltzmannStrategy(SelectStrategy):
    def __init__(self, temperature: float):
        self.temperature = temperature
    
    def select(self, q_table: np.ndarray, state: int, actions_num: int) -> int:
        q_values = q_table[state]
        exp_q = np.exp(q_values / self.temperature)
        probabilities = exp_q / np.sum(exp_q)
        action = np.random.choice(actions_num, p=probabilities)
        return action
    
    def __str__(self) -> str:
        return 'Boltzmann Strategy'
    
class CountBasedStrategy(SelectStrategy):
    def __init__(self, actions_num: int, states_num: int):
        self.visit_count = np.zeros((states_num, actions_num))

    def select(self, q_table: np.ndarray, state: int, actions_num: int) -> int:
        exploration_bonus = 1 / (np.sqrt(self.visit_count[state] + 1))
        adjusted_q_values = q_table[state] + exploration_bonus
        action = np.argmax(adjusted_q_values)
        self.visit_count[state, action] += 1
        return action

    def __str__(self) -> str:
        return 'Count Based Strategy'
    
@dataclass
class AgentParams:
    env: Environment
    learning_rate: float = 0.1
    discount_factor: float = 0.99

class QLearningAgent:
    def __init__(self, params: AgentParams):
        self.learning_rate = params.learning_rate
        self.discount_factor = params.discount_factor
        self.env = params.env
        self.states_num = params.env.states_num
        self.actions_num = params.env.actions_num
        self.q_table = np.zeros((self.states_num, self.actions_num))
        
    def update_q(self, state: int, action: int, reward: int, next_state: int, done: bool) -> None:
        max_q_next = np.max(self.q_table[next_state]) if not done else 0
        self.q_table[state, action] = self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.discount_factor * max_q_next - self.q_table[state, action])
        
    def train(self, episodes: int, strategy: SelectStrategy) -> List[int]:
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()  # Reset środowiska
            total_reward = 0
            done = False

            while not done:
                action = strategy.select(self.q_table, state, self.actions_num)  # Wybór akcji
                next_state, reward, done = self.env.step(action)  # Wykonanie akcji
                self.update_q(state, action, reward, next_state, done)  # Aktualizacja Q
                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        return rewards