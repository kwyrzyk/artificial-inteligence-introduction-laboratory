from QLearningAgent import *
import gym
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def test_learning_rates(episodes, trials, learning_rates, strategy):
    results = {lr: [] for lr in learning_rates}

    environments = []
    for _ in range(trials):
        env = GymEnvironment('Taxi-v3')
        environments.append(env)

    for lr in learning_rates:
        trial_rewards = []
        for env in environments:
            env_copy = copy.deepcopy(env)
            agent_params = AgentParams(env_copy, learning_rate=lr, discount_factor=0.99)
            agent = QLearningAgent(agent_params)
            rewards = agent.train(episodes, strategy)
            trial_rewards.append(rewards)
        
        avg_rewards = np.mean(trial_rewards, axis=0)
        results[lr] = avg_rewards   

    plt.figure(figsize=(10, 6))
    for lr, rewards in results.items():
        plt.plot(range(episodes), rewards, label=f"Learning Rate: {lr}")

    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.title("Comparison of Learning Rates - " + str(strategy))
    plt.legend()
    plt.grid()
    plt.savefig(str(strategy).replace(" ", "_").lower() + '_learning_rate.png')

def test_strategies(episodes, trials, learning_rate, strategies):
    results = {str(strategy): [] for strategy in strategies}
    envs = []

    for _ in range(trials):
        env = GymEnvironment('Taxi-v3')
        envs.append(env)

    for strategy in strategies:
        trial_rewards = []
        for env in envs:
            env_copy = copy.deepcopy(env)
            agent_params = AgentParams(env_copy, learning_rate=learning_rate, discount_factor=0.99)
            agent = QLearningAgent(agent_params)
            rewards = agent.train(episodes, strategy)
            trial_rewards.append(rewards)

        avg_rewards = np.mean(trial_rewards, axis=0)
        results[str(strategy)] = avg_rewards

    plt.figure(figsize=(12, 8))
    for strategy, rewards in results.items():
        plt.plot(range(episodes), rewards, label=str(strategy))

    plt.xlabel("Episode")
    plt.ylabel("Average Total Reward")
    plt.title(f"Comparison of Exploration Strategies (Learning Rate = {learning_rate})")
    plt.legend()
    plt.grid()
    plt.savefig(f"strategies_comparison_lr_{learning_rate}.png")

class GymEnvironment(Environment):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.states_num = self.env.observation_space.n
        self.actions_num = self.env.action_space.n

    def reset(self):
        state, _ = self.env.reset()
        return int(state)
    
    def step(self, action):
        next_state, reward, done,_ ,_ = self.env.step(action)
        return int(next_state), int(reward), int(done)

if __name__ == "__main__":


    EPISODES = 1000
    TRIALS = 5
    LEARNING_RATE = 0.9
    STRATEGIES = [
        EpsilonGreedyStrategy(epsilon=0.1),
        BoltzmannStrategy(temperature=1.0),
        CountBasedStrategy(actions_num=6, states_num=500)
    ]
    LEARNING_RATES = [0.01, 0.05, 0.1, 0.5, 1]

    test_learning_rates(EPISODES, TRIALS, LEARNING_RATES, STRATEGIES[0])
    test_learning_rates(EPISODES, TRIALS, LEARNING_RATES, STRATEGIES[1])
    test_learning_rates(EPISODES, TRIALS, LEARNING_RATES, STRATEGIES[2])
    test_strategies(EPISODES, TRIALS, LEARNING_RATE, STRATEGIES)